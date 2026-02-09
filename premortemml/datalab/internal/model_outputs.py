from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelOutput(ABC):

    data: np.ndarray

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def collect(self):
        pass

class MultiClassPredProbs(ModelOutput):

    argument = "pred_probs"

    def validate(self):
        pred_probs = self.data
        if pred_probs.ndim != 2:
            raise ValueError("pred_probs must be a 2D array for multi-class classification")
        if not np.all((pred_probs >= 0) & (pred_probs <= 1)):
            incorrect_range = (np.min(pred_probs), np.max(pred_probs))
            raise ValueError(
                "Expected pred_probs to be between 0 and 1 for multi-label classification,"
                f" but got values in range {incorrect_range} instead."
            )
        if not np.allclose(np.sum(pred_probs, axis=1), 1):
            raise ValueError("pred_probs must sum to 1 for each row for multi-class classification")

    def collect(self):
        return self.data

class RegressionPredictions(ModelOutput):

    argument = "predictions"

    def validate(self):
        predictions = self.data
        if predictions.ndim != 1:
            raise ValueError("pred_probs must be a 1D array for regression")

    def collect(self):
        return self.data

class MultiLabelPredProbs(ModelOutput):

    argument = "pred_probs"

    def validate(self):
        pred_probs = self.data
        if pred_probs.ndim != 2:
            raise ValueError(
                f"Expected pred_probs to be a 2D array for multi-label classification,"
                " but got {pred_probs.ndim}D array instead."
            )
        if not np.all((pred_probs >= 0) & (pred_probs <= 1)):
            incorrect_range = (np.min(pred_probs), np.max(pred_probs))
            raise ValueError(
                "Expected pred_probs to be between 0 and 1 for multi-label classification,"
                f" but got values in range {incorrect_range} instead."
            )

    def collect(self):
        return self.data
