from sklearn import metrics
from utils.dtypes import *


def roc(novelty_scores: list, labels: list) -> Tuple[list, list, list, float]:
    fpr, tpr, thresholds = metrics.roc_curve(labels, novelty_scores)
    roc_auc_score = metrics.roc_auc_score(labels, novelty_scores)
    return fpr, tpr, thresholds, roc_auc_score


def precision_at_k(novelty_scores: list, labels: list) -> list:
    pak = []
    n_tp = 0
    k = 1

    scores_arg_sort_asc = np.argsort(novelty_scores)
    for i in scores_arg_sort_asc[::-1]:
        if labels[i] > 0.99:
            n_tp += 1
        pak.append(n_tp / k)
        k += 1
    return pak


def iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : np.array w/ length 4
    bb2 : np.array w/ length 4

    Both bounding boxes have the following format:
        The coordinate (bb[0], bb[1]) is the position at the top left corner of
        the bbox (e.g. x_left, y_top), bb[2] and bb[3] are the width and height
        of the bbox respectively

    Returns
    -------
    float
        in [0, 1]
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
    y_bottom = min(bb1[1]+bb1[3], bb2[1]+bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both AABBs
    bb1_area = (bb1[2]) * (bb1[3])
    bb2_area = (bb2[2]) * (bb2[3])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
