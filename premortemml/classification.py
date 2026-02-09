from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import inspect
import warnings
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from typing_extensions import Self

from premortemml.rank import get_label_quality_scores
from premortemml.internal.util import (
    value_counts,
    compress_int_array,
    subset_X_y,
    get_num_classes,
    force_two_dimensions,
)
from premortemml.count import (
    estimate_py_noise_matrices_and_cv_pred_proba,
    estimate_py_and_noise_matrices_from_probabilities,
    estimate_cv_predicted_probabilities,
    estimate_latent,
    compute_confident_joint,
)
from premortemml.internal.latent_algebra import (
    compute_py_inv_noise_matrix,
    compute_noise_matrix_from_inverse,
)
from premortemml.internal.validation import (
    assert_valid_inputs,
    labels_to_array,
)
from premortemml.experimental.label_issues_batched import find_label_issues_batched

class CleanLearning(BaseEstimator):  # Inherits sklearn classifier

    def __init__(
        self,
        clf=None,
        *,
        seed=None,
        # Hyper-parameters (used by .fit() function)
        cv_n_folds=5,
        converge_latent_estimates=False,
        pulearning=None,
        find_label_issues_kwargs={},
        label_quality_scores_kwargs={},
        verbose=False,
        low_memory=False,
    ):
        self._default_clf = False
        if clf is None:
            # Use logistic regression if no classifier is provided.
            clf = LogReg(solver="lbfgs")
            self._default_clf = True

        # Make sure the given classifier has the appropriate methods defined.
        if not hasattr(clf, "fit"):
            raise ValueError("The classifier (clf) must define a .fit() method.")
        if not hasattr(clf, "predict_proba"):
            raise ValueError("The classifier (clf) must define a .predict_proba() method.")
        if not hasattr(clf, "predict"):
            raise ValueError("The classifier (clf) must define a .predict() method.")

        if seed is not None:
            np.random.seed(seed=seed)

        self.clf = clf
        self.seed = seed
        self.cv_n_folds = cv_n_folds
        self.converge_latent_estimates = converge_latent_estimates
        self.pulearning = pulearning
        self.find_label_issues_kwargs = find_label_issues_kwargs
        self.label_quality_scores_kwargs = label_quality_scores_kwargs
        self.verbose = verbose
        self.label_issues_df = None
        self.label_issues_mask = None
        self.sample_weight = None
        self.confident_joint = None
        self.py = None
        self.ps = None
        self.num_classes = None
        self.noise_matrix = None
        self.inverse_noise_matrix = None
        self.clf_kwargs = None
        self.clf_final_kwargs = None
        self.low_memory = low_memory

    def fit(
        self,
        X,
        labels=None,
        *,
        pred_probs=None,
        thresholds=None,
        noise_matrix=None,
        inverse_noise_matrix=None,
        label_issues=None,
        sample_weight=None,
        clf_kwargs={},
        clf_final_kwargs={},
        validation_func=None,
        y=None,
    ) -> "Self":
        

        if labels is not None and y is not None:
            raise ValueError("You must specify either `labels` or `y`, but not both.")
        if y is not None:
            labels = y
        if labels is None:
            raise ValueError("You must specify `labels`.")
        if self._default_clf:
            X = force_two_dimensions(X)

        self.clf_final_kwargs = {**clf_kwargs, **clf_final_kwargs}

        if "sample_weight" in clf_kwargs:
            raise ValueError(
                "sample_weight should be provided directly in fit() or in clf_final_kwargs rather than in clf_kwargs"
            )

        if sample_weight is not None:
            if "sample_weight" not in inspect.signature(self.clf.fit).parameters:
                raise ValueError(
                    "sample_weight must be a supported fit() argument for your model in order to be specified here"
                )

        if label_issues is None:
            if self.label_issues_df is not None and self.verbose:
                print(
                    "If you already ran self.find_label_issues() and don't want to recompute, you "
                    "should pass the label_issues in as a parameter to this function next time."
                )
            label_issues = self.find_label_issues(
                X,
                labels,
                pred_probs=pred_probs,
                thresholds=thresholds,
                noise_matrix=noise_matrix,
                inverse_noise_matrix=inverse_noise_matrix,
                clf_kwargs=clf_kwargs,
                validation_func=validation_func,
            )

        else:  # set args that may not have been set if `self.find_label_issues()` wasn't called yet
            assert_valid_inputs(X, labels, pred_probs)
            if self.num_classes is None:
                if noise_matrix is not None:
                    label_matrix = noise_matrix
                else:
                    label_matrix = inverse_noise_matrix
                self.num_classes = get_num_classes(labels, pred_probs, label_matrix)
            if self.verbose:
                print("Using provided label_issues instead of finding label issues.")
                if self.label_issues_df is not None:
                    print(
                        "These will overwrite self.label_issues_df and will be returned by "
                        "`self.get_label_issues()`. "
                    )

        # label_issues always overwrites self.label_issues_df. Ensure it is properly formatted:
        self.label_issues_df = self._process_label_issues_arg(label_issues, labels)

        if "label_quality" not in self.label_issues_df.columns and pred_probs is not None:
            if self.verbose:
                print("Computing label quality scores based on given pred_probs ...")
            self.label_issues_df["label_quality"] = get_label_quality_scores(
                labels, pred_probs, **self.label_quality_scores_kwargs
            )

        self.label_issues_mask = self.label_issues_df["is_label_issue"].to_numpy()
        x_mask = np.invert(self.label_issues_mask)
        x_cleaned, labels_cleaned = subset_X_y(X, labels, x_mask)
        if self.verbose:
            print(f"Pruning {np.sum(self.label_issues_mask)} examples with label issues ...")
            print(f"Remaining clean data has {len(labels_cleaned)} examples.")

        if sample_weight is None:
            # Check if sample_weight in args of clf.fit()
            if (
                "sample_weight" in inspect.signature(self.clf.fit).parameters
                and "sample_weight" not in self.clf_final_kwargs
                and self.noise_matrix is not None
            ):
                # Re-weight examples in the loss function for the final fitting
                # such that the "apparent" original number of examples in each class
                # is preserved, even though the pruned sets may differ.
                if self.verbose:
                    print(
                        "Assigning sample weights for final training based on estimated label quality."
                    )
                sample_weight_auto = np.ones(np.shape(labels_cleaned))
                for k in range(self.num_classes):
                    sample_weight_k = 1.0 / max(
                        self.noise_matrix[k][k], 1e-3
                    )  # clip sample weights
                    sample_weight_auto[labels_cleaned == k] = sample_weight_k

                sample_weight_expanded = np.zeros(
                    len(labels)
                )  # pad pruned examples with zeros, length of original dataset
                sample_weight_expanded[x_mask] = sample_weight_auto
                # Store the sample weight for every example in the original, unfiltered dataset
                self.label_issues_df["sample_weight"] = sample_weight_expanded
                self.sample_weight = self.label_issues_df[
                    "sample_weight"
                ]  # pointer to here to avoid duplication
                self.clf_final_kwargs["sample_weight"] = sample_weight_auto
                if self.verbose:
                    print("Fitting final model on the clean data ...")
            else:
                if self.verbose:
                    if "sample_weight" in self.clf_final_kwargs:
                        print("Fitting final model on the clean data with custom sample_weight ...")
                    else:
                        if (
                            "sample_weight" in inspect.signature(self.clf.fit).parameters
                            and self.noise_matrix is None
                        ):
                            print(
                                "Cannot utilize sample weights for final training! "
                                "Why this matters: during final training, sample weights help account for the amount of removed data in each class. "
                                "This helps ensure the correct class prior for the learned model. "
                                "To use sample weights, you need to either provide the noise_matrix or have previously called self.find_label_issues() instead of filter.find_label_issues() which computes them for you."
                            )
                        print("Fitting final model on the clean data ...")

        elif sample_weight is not None and "sample_weight" not in self.clf_final_kwargs:
            self.clf_final_kwargs["sample_weight"] = sample_weight[x_mask]
            if self.verbose:
                print("Fitting final model on the clean data with custom sample_weight ...")

        else:  # pragma: no cover
            if self.verbose:
                if "sample_weight" in self.clf_final_kwargs:
                    print("Fitting final model on the clean data with custom sample_weight ...")
                else:
                    print("Fitting final model on the clean data ...")

        self.clf.fit(x_cleaned, labels_cleaned, **self.clf_final_kwargs)

        if self.verbose:
            print(
                "Label issues stored in label_issues_df DataFrame accessible via: self.get_label_issues(). "
                "Call self.save_space() to delete this potentially large DataFrame attribute."
            )
        return self

    def predict(self, *args, **kwargs) -> np.ndarray:
        if self._default_clf:
            if args:
                X = args[0]
            elif "X" in kwargs:
                X = kwargs["X"]
                del kwargs["X"]
            else:
                raise ValueError("No input provided to predict, please provide X.")
            X = force_two_dimensions(X)
            new_args = (X,) + args[1:]
            return self.clf.predict(*new_args, **kwargs)
        else:
            return self.clf.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        if self._default_clf:
            if args:
                X = args[0]
            elif "X" in kwargs:
                X = kwargs["X"]
                del kwargs["X"]
            else:
                raise ValueError("No input provided to predict, please provide X.")
            X = force_two_dimensions(X)
            new_args = (X,) + args[1:]
            return self.clf.predict_proba(*new_args, **kwargs)
        else:
            return self.clf.predict_proba(*args, **kwargs)

    def score(self, X, y, sample_weight=None) -> float:
        if self._default_clf:
            X = force_two_dimensions(X)
        if hasattr(self.clf, "score"):
            # Check if sample_weight in clf.score()
            if "sample_weight" in inspect.signature(self.clf.score).parameters:
                return self.clf.score(X, y, sample_weight=sample_weight)
            else:
                return self.clf.score(X, y)
        else:
            return accuracy_score(
                y,
                self.clf.predict(X),
                sample_weight=sample_weight,
            )

    def find_label_issues(
        self,
        X=None,
        labels=None,
        *,
        pred_probs=None,
        thresholds=None,
        noise_matrix=None,
        inverse_noise_matrix=None,
        save_space=False,
        clf_kwargs={},
        validation_func=None,
    ) -> pd.DataFrame:
        

        # Check inputs
        assert_valid_inputs(X, labels, pred_probs)
        labels = labels_to_array(labels)
        if noise_matrix is not None and np.trace(noise_matrix) <= 1:
            t = np.round(np.trace(noise_matrix), 2)
            raise ValueError("Trace(noise_matrix) is {}, but must exceed 1.".format(t))
        if inverse_noise_matrix is not None and (np.trace(inverse_noise_matrix) <= 1):
            t = np.round(np.trace(inverse_noise_matrix), 2)
            raise ValueError("Trace(inverse_noise_matrix) is {}. Must exceed 1.".format(t))

        if self._default_clf:
            X = force_two_dimensions(X)
        if noise_matrix is not None:
            label_matrix = noise_matrix
        else:
            label_matrix = inverse_noise_matrix
        self.num_classes = get_num_classes(labels, pred_probs, label_matrix)
        if (pred_probs is None) and (len(labels) / self.num_classes < self.cv_n_folds):
            raise ValueError(
                "Need more data from each class for cross-validation. "
                "Try decreasing cv_n_folds (eg. to 2 or 3) in CleanLearning()"
            )
        # 'ps' is p(labels=k)
        self.ps = value_counts(labels) / float(len(labels))

        self.clf_kwargs = clf_kwargs
        if self.low_memory:
            # If needed, compute P(label=k|x), denoted pred_probs (the predicted probabilities)
            if pred_probs is None:
                if self.verbose:
                    print(
                        "Computing out of sample predicted probabilities via "
                        f"{self.cv_n_folds}-fold cross validation. May take a while ..."
                    )

                pred_probs = estimate_cv_predicted_probabilities(
                    X=X,
                    labels=labels,
                    clf=self.clf,
                    cv_n_folds=self.cv_n_folds,
                    seed=self.seed,
                    clf_kwargs=self.clf_kwargs,
                    validation_func=validation_func,
                )

            if self.verbose:
                print("Using predicted probabilities to identify label issues ...")

            if self.find_label_issues_kwargs:
                warnings.warn(f"`find_label_issues_kwargs` is not used when `low_memory=True`.")
            arg_values = {
                "thresholds": thresholds,
                "noise_matrix": noise_matrix,
                "inverse_noise_matrix": inverse_noise_matrix,
            }
            for arg_name, arg_val in arg_values.items():
                if arg_val is not None:
                    warnings.warn(f"`{arg_name}` is not used when `low_memory=True`.")
            label_issues_mask = find_label_issues_batched(labels, pred_probs, return_mask=True)
        else:
            self._process_label_issues_kwargs(self.find_label_issues_kwargs)
            # self._process_label_issues_kwargs might set self.confident_joint. If so, we should use it.
            if self.confident_joint is not None:
                self.py, noise_matrix, inv_noise_matrix = estimate_latent(
                    confident_joint=self.confident_joint,
                    labels=labels,
                )

            # If needed, compute noise rates (probability of class-conditional mislabeling).
            if noise_matrix is not None:
                self.noise_matrix = noise_matrix
                if inverse_noise_matrix is None:
                    if self.verbose:
                        print("Computing label noise estimates from provided noise matrix ...")
                    self.py, self.inverse_noise_matrix = compute_py_inv_noise_matrix(
                        ps=self.ps,
                        noise_matrix=self.noise_matrix,
                    )
            if inverse_noise_matrix is not None:
                self.inverse_noise_matrix = inverse_noise_matrix
                if noise_matrix is None:
                    if self.verbose:
                        print(
                            "Computing label noise estimates from provided inverse noise matrix ..."
                        )
                    self.noise_matrix = compute_noise_matrix_from_inverse(
                        ps=self.ps,
                        inverse_noise_matrix=self.inverse_noise_matrix,
                    )

            if noise_matrix is None and inverse_noise_matrix is None:
                if pred_probs is None:
                    if self.verbose:
                        print(
                            "Computing out of sample predicted probabilities via "
                            f"{self.cv_n_folds}-fold cross validation. May take a while ..."
                        )
                    (
                        self.py,
                        self.noise_matrix,
                        self.inverse_noise_matrix,
                        self.confident_joint,
                        pred_probs,
                    ) = estimate_py_noise_matrices_and_cv_pred_proba(
                        X=X,
                        labels=labels,
                        clf=self.clf,
                        cv_n_folds=self.cv_n_folds,
                        thresholds=thresholds,
                        converge_latent_estimates=self.converge_latent_estimates,
                        seed=self.seed,
                        clf_kwargs=self.clf_kwargs,
                        validation_func=validation_func,
                    )
                else:  # pred_probs is provided by user (assumed holdout probabilities)
                    if self.verbose:
                        print("Computing label noise estimates from provided pred_probs ...")
                    (
                        self.py,
                        self.noise_matrix,
                        self.inverse_noise_matrix,
                        self.confident_joint,
                    ) = estimate_py_and_noise_matrices_from_probabilities(
                        labels=labels,
                        pred_probs=pred_probs,
                        thresholds=thresholds,
                        converge_latent_estimates=self.converge_latent_estimates,
                    )
            # If needed, compute P(label=k|x), denoted pred_probs (the predicted probabilities)
            if pred_probs is None:
                if self.verbose:
                    print(
                        "Computing out of sample predicted probabilities via "
                        f"{self.cv_n_folds}-fold cross validation. May take a while ..."
                    )

                pred_probs = estimate_cv_predicted_probabilities(
                    X=X,
                    labels=labels,
                    clf=self.clf,
                    cv_n_folds=self.cv_n_folds,
                    seed=self.seed,
                    clf_kwargs=self.clf_kwargs,
                    validation_func=validation_func,
                )
            # If needed, compute the confident_joint (e.g. occurs if noise_matrix was given)
            if self.confident_joint is None:
                self.confident_joint = compute_confident_joint(
                    labels=labels,
                    pred_probs=pred_probs,
                    thresholds=thresholds,
                )

            # if pulearning == the integer specifying the class without noise.
            if self.num_classes == 2 and self.pulearning is not None:  # pragma: no cover
                # pulearning = 1 (no error in 1 class) implies p(label=1|true_label=0) = 0
                self.noise_matrix[self.pulearning][1 - self.pulearning] = 0
                self.noise_matrix[1 - self.pulearning][1 - self.pulearning] = 1
                # pulearning = 1 (no error in 1 class) implies p(true_label=0|label=1) = 0
                self.inverse_noise_matrix[1 - self.pulearning][self.pulearning] = 0
                self.inverse_noise_matrix[self.pulearning][self.pulearning] = 1
                # pulearning = 1 (no error in 1 class) implies p(label=1,true_label=0) = 0
                self.confident_joint[self.pulearning][1 - self.pulearning] = 0
                self.confident_joint[1 - self.pulearning][1 - self.pulearning] = 1

            # Add confident joint to find label issue args if it is not previously specified
            if "confident_joint" not in self.find_label_issues_kwargs.keys():
                # however does not add if users specify filter_by="confident_learning", as it will throw a warning
                if not self.find_label_issues_kwargs.get("filter_by") == "confident_learning":
                    self.find_label_issues_kwargs["confident_joint"] = self.confident_joint

            labels = labels_to_array(labels)
            if self.verbose:
                print("Using predicted probabilities to identify label issues ...")
            label_issues_mask = filter.find_label_issues(
                labels,
                pred_probs,
                **self.find_label_issues_kwargs,
            )
        label_quality_scores = get_label_quality_scores(
            labels, pred_probs, **self.label_quality_scores_kwargs
        )
        label_issues_df = pd.DataFrame(
            {"is_label_issue": label_issues_mask, "label_quality": label_quality_scores}
        )
        if self.verbose:
            print(f"Identified {np.sum(label_issues_mask)} examples with label issues.")

        predicted_labels = pred_probs.argmax(axis=1)
        label_issues_df["given_label"] = compress_int_array(labels, self.num_classes)
        label_issues_df["predicted_label"] = compress_int_array(predicted_labels, self.num_classes)

        if not save_space:
            if self.label_issues_df is not None and self.verbose:
                print(
                    "Overwriting previously identified label issues stored at self.label_issues_df. "
                    "self.get_label_issues() will now return the newly identified label issues. "
                )
            self.label_issues_df = label_issues_df
            self.label_issues_mask = label_issues_df[
                "is_label_issue"
            ]  # pointer to here to avoid duplication
        elif self.verbose:
            print(  # pragma: no cover
                "Not storing label_issues as attributes since save_space was specified."
            )

        return label_issues_df

    def get_label_issues(self) -> Optional[pd.DataFrame]:

        if self.label_issues_df is None:
            warnings.warn(
                "Label issues have not yet been computed. Run `self.find_label_issues()` or `self.fit()` first."
            )
        return self.label_issues_df

    def save_space(self):

        if self.label_issues_df is None and self.verbose:
            print("self.label_issues_df is already empty")  # pragma: no cover
        self.label_issues_df = None
        self.sample_weight = None
        self.label_issues_mask = None
        self.find_label_issues_kwargs = None
        self.label_quality_scores_kwargs = None
        self.confident_joint = None
        self.py = None
        self.ps = None
        self.num_classes = None
        self.noise_matrix = None
        self.inverse_noise_matrix = None
        self.clf_kwargs = None
        self.clf_final_kwargs = None
        if self.verbose:
            print("Deleted non-sklearn attributes such as label_issues_df to save space.")

    def _process_label_issues_kwargs(self, find_label_issues_kwargs):

        # Defaults for CleanLearning.find_label_issues() vs filter.find_label_issues()
        DEFAULT_FIND_LABEL_ISSUES_KWARGS = {"min_examples_per_class": 10}
        find_label_issues_kwargs = {**DEFAULT_FIND_LABEL_ISSUES_KWARGS, **find_label_issues_kwargs}
        unsupported_kwargs = ["return_indices_ranked_by", "multi_label"]
        for unsupported_kwarg in unsupported_kwargs:
            if unsupported_kwarg in find_label_issues_kwargs:
                raise ValueError(
                    "These kwargs of `find_label_issues()` are not supported "
                    f"for `CleanLearning`: {unsupported_kwargs}"
                )
        # CleanLearning will use this to compute the noise_matrix and inverse_noise_matrix
        if "confident_joint" in find_label_issues_kwargs:
            self.confident_joint = find_label_issues_kwargs["confident_joint"]
        self.find_label_issues_kwargs = find_label_issues_kwargs

    def _process_label_issues_arg(self, label_issues, labels) -> pd.DataFrame:

        labels = labels_to_array(labels)
        if isinstance(label_issues, pd.DataFrame):
            if "is_label_issue" not in label_issues.columns:
                raise ValueError(
                    "DataFrame label_issues must contain column: 'is_label_issue'. "
                    "See CleanLearning.fit() documentation for label_issues column descriptions."
                )
            if len(label_issues) != len(labels):
                raise ValueError("label_issues and labels must have same length")
            if "given_label" in label_issues.columns and np.any(
                label_issues["given_label"].to_numpy() != labels
            ):
                raise ValueError("labels must match label_issues['given_label']")
            return label_issues
        elif isinstance(label_issues, np.ndarray):
            if not label_issues.dtype in [np.dtype("bool"), np.dtype("int")]:
                raise ValueError("If label_issues is numpy.array, dtype must be 'bool' or 'int'.")
            if label_issues.dtype is np.dtype("bool") and label_issues.shape != labels.shape:
                raise ValueError(
                    "If label_issues is boolean numpy.array, must have same shape as labels"
                )
            if label_issues.dtype is np.dtype("int"):  # convert to boolean mask
                if len(np.unique(label_issues)) != len(label_issues):
                    raise ValueError(
                        "If label_issues.dtype is 'int', must contain unique integer indices "
                        "corresponding to examples with label issues such as output by: "
                        "filter.find_label_issues(..., return_indices_ranked_by=...)"
                    )
                issue_indices = label_issues
                label_issues = np.full(len(labels), False, dtype=bool)
                if len(issue_indices) > 0:
                    label_issues[issue_indices] = True
            return pd.DataFrame({"is_label_issue": label_issues})
        else:
            raise ValueError("label_issues must be either pandas.DataFrame or numpy.array")
