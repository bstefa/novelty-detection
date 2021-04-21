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
