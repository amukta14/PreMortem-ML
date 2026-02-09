from multiprocessing import Pool
from typing import (
    Optional,
    Any,
    Dict,
    Tuple,
    Union,
    List,
    TYPE_CHECKING,
    TypeVar,
    DefaultDict,
    cast,
)

import numpy as np
import collections

from premortemml.internal.constants import (
    MAX_CLASS_TO_SHOW,
    ALPHA,
    EPSILON,
    TINY_VALUE,
)
from premortemml.object_detection.filter import (
    _filter_by_class,
    _calculate_true_positives_false_positives,
)
from premortemml.object_detection.rank import (
    _get_valid_inputs_for_compute_scores,
    _separate_prediction,
    _separate_label,
    _get_prediction_type,
)

from premortemml.internal.object_detection_utils import bbox_xyxy_to_xywh

if TYPE_CHECKING:
    from PIL.Image import Image as Image  # pragma: no cover
else:
    Image = TypeVar("Image")

def object_counts_per_image(
    labels=None,
    predictions=None,
    *,
    auxiliary_inputs=None,
) -> Tuple[List, List]:
    
    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)
    return (
        [len(sample["lab_bboxes"]) for sample in auxiliary_inputs],
        [len(sample["pred_bboxes"]) for sample in auxiliary_inputs],
    )

def bounding_box_size_distribution(
    labels=None,
    predictions=None,
    *,
    auxiliary_inputs=None,
    class_names: Optional[Dict[Any, Any]] = None,
    sort: bool = False,
) -> Tuple[Dict[Any, List], Dict[Any, List]]:
    
    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)

    lab_area: Dict[Any, list] = collections.defaultdict(list)
    pred_area: Dict[Any, list] = collections.defaultdict(list)
    for sample in auxiliary_inputs:
        _get_bbox_areas(sample["lab_labels"], sample["lab_bboxes"], lab_area, class_names)
        _get_bbox_areas(sample["pred_labels"], sample["pred_bboxes"], pred_area, class_names)

    if sort:
        lab_area = dict(sorted(lab_area.items(), key=lambda x: -len(x[1])))
        pred_area = dict(sorted(pred_area.items(), key=lambda x: -len(x[1])))

    return lab_area, pred_area

def class_label_distribution(
    labels=None,
    predictions=None,
    *,
    auxiliary_inputs=None,
    class_names: Optional[Dict[Any, Any]] = None,
) -> Tuple[Dict[Any, float], Dict[Any, float]]:
    
    if auxiliary_inputs is None:
        auxiliary_inputs = _get_valid_inputs_for_compute_scores(ALPHA, labels, predictions)

    lab_freq: DefaultDict[Any, int] = collections.defaultdict(int)
    pred_freq: DefaultDict[Any, int] = collections.defaultdict(int)
    for sample in auxiliary_inputs:
        _get_class_instances(sample["lab_labels"], lab_freq, class_names)
        _get_class_instances(sample["pred_labels"], pred_freq, class_names)

    label_norm = _normalize_by_total(lab_freq)
    pred_norm = _normalize_by_total(pred_freq)

    return label_norm, pred_norm

def get_sorted_bbox_count_idxs(labels, predictions):
    lab_count, pred_count = object_counts_per_image(labels, predictions)
    lab_grouped = list(enumerate(lab_count))
    pred_grouped = list(enumerate(pred_count))

    sorted_lab = sorted(lab_grouped, key=lambda x: x[1], reverse=True)
    sorted_pred = sorted(pred_grouped, key=lambda x: x[1], reverse=True)

    return sorted_lab, sorted_pred

def plot_class_size_distributions(
    labels, predictions, class_names=None, class_to_show=MAX_CLASS_TO_SHOW, **kwargs
):
    
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "This functionality requires matplotlib. Install it via: `pip install matplotlib`"
        )

    lab_boxes, pred_boxes = bounding_box_size_distribution(
        labels,
        predictions,
        class_names=class_names,
        sort=True if class_to_show is not None else False,
    )

    for i, c in enumerate(lab_boxes.keys()):
        if i >= class_to_show:
            break
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Size distributions for bounding box for class {c}")
        for i, l in enumerate([lab_boxes, pred_boxes]):
            axs[i].hist(l[c], bins="auto")
            axs[i].set_xlabel("box area (pixels)")
            axs[i].set_ylabel("count")
            axs[i].set_title("annotated" if i == 0 else "predicted")

        plt.show(**kwargs)

def plot_class_distribution(labels, predictions, class_names=None, **kwargs):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "This functionality requires matplotlib. Install it via: `pip install matplotlib`"
        )

    lab_dist, pred_dist = class_label_distribution(labels, predictions, class_names=class_names)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Distribution of classes in the dataset")
    for i, d in enumerate([lab_dist, pred_dist]):
        axs[i].pie(d.values(), labels=d.keys(), autopct="%1.1f%%")
        axs[i].set_title("Annotated" if i == 0 else "Predicted")

    plt.show(**kwargs)

def visualize(
    image: Union[str, np.ndarray, Image],
    *,
    label: Optional[Dict[str, Any]] = None,
    prediction: Optional[np.ndarray] = None,
    prediction_threshold: Optional[float] = None,
    overlay: bool = True,
    class_names: Optional[Dict[Any, Any]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    **kwargs,
) -> None:
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
    except ImportError as e:
        raise ImportError(
            "This functionality requires matplotlib. Install it via: `pip install matplotlib`"
        )

    # Create figure and axes
    if isinstance(image, str):
        image = plt.imread(image)

    if prediction is not None:
        prediction_type = _get_prediction_type(prediction)
        pbbox, plabels, pred_probs = _separate_prediction(
            prediction, prediction_type=prediction_type
        )

        if prediction_threshold is not None:
            keep_idx = np.where(pred_probs > prediction_threshold)
            pbbox = pbbox[keep_idx]
            plabels = plabels[keep_idx]

    if label is not None:
        abbox, alabels = _separate_label(label)

    if overlay:
        figsize = (8, 5) if figsize is None else figsize
        fig, ax = plt.subplots(frameon=False, figsize=figsize)
        plt.axis("off")
        ax.imshow(image)
        if label is not None:
            fig, ax = _draw_boxes(
                fig, ax, abbox, alabels, edgecolor="r", linestyle="-", linewidth=1
            )
        if prediction is not None:
            _, _ = _draw_boxes(fig, ax, pbbox, plabels, edgecolor="b", linestyle="-.", linewidth=1)
    else:
        figsize = (14, 10) if figsize is None else figsize
        fig, axes = plt.subplots(nrows=1, ncols=2, frameon=False, figsize=figsize)
        axes = cast(Tuple[Axes, Axes], axes)
        axes[0].axis("off")
        axes[0].imshow(image)
        axes[1].axis("off")
        axes[1].imshow(image)

        if label is not None:
            fig, ax = _draw_boxes(
                fig, axes[0], abbox, alabels, edgecolor="r", linestyle="-", linewidth=1
            )
        if prediction is not None:
            _, _ = _draw_boxes(
                fig, axes[1], pbbox, plabels, edgecolor="b", linestyle="-.", linewidth=1
            )
    bbox_extra_artists = None
    if label or prediction is not None:
        legend, plt = _plot_legend(class_names, label, prediction)
        bbox_extra_artists = (legend,)

    if save_path:
        allowed_image_formats = set(["png", "pdf", "ps", "eps", "svg"])
        image_format: Optional[str] = None
        if save_path.split(".")[-1] in allowed_image_formats and "." in save_path:
            image_format = save_path.split(".")[-1]
        plt.savefig(
            save_path,
            format=image_format,
            bbox_extra_artists=bbox_extra_artists,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.5,
        )
    plt.show(**kwargs)

def _get_per_class_confusion_matrix_dict_(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    iou_threshold: Optional[float] = 0.5,
    num_procs: int = 1,
) -> DefaultDict[int, Dict[str, int]]:
    
    num_classes = len(predictions[0])
    num_images = len(predictions)
    pool = Pool(num_procs)
    counter_dict: DefaultDict[int, dict[str, int]] = collections.defaultdict(
        lambda: {"TP": 0, "FP": 0, "FN": 0}
    )

    for class_num in range(num_classes):
        pred_bboxes, lab_bboxes = _filter_by_class(labels, predictions, class_num)
        tpfpfn = pool.starmap(
            _calculate_true_positives_false_positives,
            zip(
                pred_bboxes,
                lab_bboxes,
                [iou_threshold for _ in range(num_images)],
                [True for _ in range(num_images)],
            ),
        )

        for image_idx, (tp, fp, fn) in enumerate(tpfpfn):  # type: ignore
            counter_dict[class_num]["TP"] += np.sum(tp)
            counter_dict[class_num]["FP"] += np.sum(fp)
            counter_dict[class_num]["FN"] += np.sum(fn)

    return counter_dict

def _sort_dict_to_list(index_value_dict):
    sorted_list = [
        value for key, value in sorted(index_value_dict.items(), key=lambda x: int(x[0]))
    ]
    return sorted_list

def get_average_per_class_confusion_matrix(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    num_procs: int = 1,
    class_names: Optional[Dict[Any, Any]] = None,
) -> Dict[Union[int, str], Dict[str, float]]:
    
    iou_thrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
    num_classes = len(predictions[0])
    if class_names is None:
        class_names = {str(i): int(i) for i in list(range(num_classes))}
    class_names = _sort_dict_to_list(class_names)
    avg_metrics = {class_num: {"TP": 0.0, "FP": 0.0, "FN": 0.0} for class_num in class_names}

    for iou_threshold in iou_thrs:
        results_dict = _get_per_class_confusion_matrix_dict_(
            labels, predictions, iou_threshold, num_procs
        )

        for class_num in results_dict:
            tp = results_dict[class_num]["TP"]
            fp = results_dict[class_num]["FP"]
            fn = results_dict[class_num]["FN"]

            avg_metrics[class_names[class_num]]["TP"] += tp
            avg_metrics[class_names[class_num]]["FP"] += fp
            avg_metrics[class_names[class_num]]["FN"] += fn

    num_thresholds = len(iou_thrs) * len(results_dict)
    for class_name in avg_metrics:
        avg_metrics[class_name]["TP"] /= num_thresholds
        avg_metrics[class_name]["FP"] /= num_thresholds
        avg_metrics[class_name]["FN"] /= num_thresholds
    return avg_metrics

def calculate_per_class_metrics(
    labels: List[Dict[str, Any]],
    predictions: List[np.ndarray],
    num_procs: int = 1,
    class_names=None,
) -> Dict[Union[int, str], Dict[str, float]]:
    
    avg_metrics = get_average_per_class_confusion_matrix(
        labels, predictions, num_procs, class_names=class_names
    )

    avg_metrics_dict = {}
    for class_name in avg_metrics:
        tp = avg_metrics[class_name]["TP"]
        fp = avg_metrics[class_name]["FP"]
        fn = avg_metrics[class_name]["FN"]

        precision = tp / (tp + fp + TINY_VALUE)  # Avoid division by zero
        recall = tp / (tp + fn + TINY_VALUE)  # Avoid division by zero
        f1 = 2 * (precision * recall) / (precision + recall + TINY_VALUE)  # Avoid division by zero

        avg_metrics_dict[class_name] = {
            "average precision": precision,
            "average recall": recall,
            "average f1": f1,
        }

    return avg_metrics_dict

def _normalize_by_total(freq):
    total = sum(freq.values())
    return {k: round(v / (total + EPSILON), 2) for k, v in freq.items()}

def _get_bbox_areas(labels, boxes, class_area_dict, class_names=None) -> None:
    for cl, bbox in zip(labels, boxes):
        if class_names is not None:
            if str(cl) not in class_names:
                continue
            cl = class_names[str(cl)]
        class_area_dict[cl].append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

def _get_class_instances(labels, class_instances_dict, class_names=None) -> None:
    for cl in labels:
        if class_names is not None:
            cl = class_names[str(cl)]
        class_instances_dict[cl] += 1

def _plot_legend(class_names, label, prediction):
    colors = ["black"]
    colors.extend(["red"] if label is not None else [])
    colors.extend(["blue"] if prediction is not None else [])

    markers = [None]
    markers.extend(["s"] if label is not None else [])
    markers.extend(["s"] if prediction is not None else [])

    labels = [r"$\bf{Legend}$"]
    labels.extend(["given label"] if label is not None else [])
    labels.extend(["predicted label"] if prediction is not None else [])

    if class_names:
        colors += ["black"] + ["black"] * min(len(class_names), MAX_CLASS_TO_SHOW)
        markers += [None] + [f"${class_key}$" for class_key in class_names.keys()]
        labels += [r"$\bf{classes}$"] + list(class_names.values())

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "This functionality requires matplotlib. Install it via: `pip install matplotlib`"
        )

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f(marker, color) for marker, color in zip(markers, colors)]
    legend = plt.legend(
        handles, labels, bbox_to_anchor=(1.04, 0.05), loc="lower left", borderaxespad=0
    )

    return legend, plt

def _draw_labels(ax, rect, label, edgecolor):

    rx, ry = rect.get_xy()
    c_xleft = rx + 10
    c_xright = rx + rect.get_width() - 10
    c_ytop = ry + 12

    if edgecolor == "r":
        cx, cy = c_xleft, c_ytop
    else:  # edgecolor == b
        cx, cy = c_xright, c_ytop

    l = ax.annotate(
        label, (cx, cy), fontsize=8, fontweight="bold", color="white", ha="center", va="center"
    )
    l.set_bbox(dict(facecolor=edgecolor, alpha=0.35, edgecolor=edgecolor, pad=2))
    return ax

def _draw_boxes(fig, ax, bboxes, labels, edgecolor="g", linestyle="-", linewidth=3):
    bboxes = [bbox_xyxy_to_xywh(box) for box in bboxes]

    try:
        from matplotlib.patches import Rectangle
    except Exception as e:
        raise ImportError(
            "This functionality requires matplotlib. Install it via: `pip install matplotlib`"
        )

    for (x, y, w, h), label in zip(bboxes, labels):
        rect = Rectangle(
            (x, y),
            w,
            h,
            linewidth=linewidth,
            linestyle=linestyle,
            edgecolor=edgecolor,
            facecolor="none",
        )
        ax.add_patch(rect)

        if labels is not None:
            ax = _draw_labels(ax, rect, label, edgecolor)

    return fig, ax
