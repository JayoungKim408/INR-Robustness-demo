from typing import Tuple, Union

import torch
from pytorch3d.ops.knn import knn_points  # type: ignore
from torch import Tensor, mean, sqrt, zeros, sum

# from pycarus.geometry.pcd import batchify, unbatchify
from basemodels.inr2vec.pcd_retrieval_and_classification.pcd import batchify, unbatchify


def chamfer(
    predictions: Tensor,
    groundtruth: Tensor,
    squared: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute the Chamfer Distance defined as in:

       Fan, H., Su, H., & Guibas, L. J. (2017).
       A point set generation network for 3d object reconstruction from a single image.
       In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 605-613).

    Args:
        prediction: The source point cloud(s) with shape ([B,] NUM_PTS_PRED, 3).
        groundtruth: The target point cloud(s) with shape ([B,] NUM_PTS_GT, 3).
        squared: If true return the squared euclidean distance.

    Returns:
        A tuple containing:
            - the chamfer distance with shape (B, ).
            - the accuracy, eg. point-wise distance from the prediction to the groundtruth with shape ([B,] NUM_PTS_PRED).
            - the completeness, eg. the distance from the groundtruth to the prediction with shape ([B,] NUM_PTS_GT).
            - the nearest neighbor indices from the prediction to the groundtruth with shape ([B,] NUM_PTS_PRED).
            - the nearest neighbor indices from the groundtruth to the prediction with shape ([B,] NUM_PTS_GT).
    """
    batched, [predictions, groundtruth] = batchify([predictions, groundtruth], 3)

    accuracy, indices_pred_gt, _ = knn_points(predictions, groundtruth)
    accuracy = accuracy.squeeze(-1)
    indices_pred_gt = indices_pred_gt.squeeze(-1)

    completeness, indices_gt_pred, _ = knn_points(groundtruth, predictions)
    completeness = completeness.squeeze(-1)
    indices_gt_pred = indices_gt_pred.squeeze(-1)

    if not squared:
        accuracy = sqrt(accuracy)
        completeness = sqrt(completeness)

    avg_accuracy = mean(accuracy, dim=-1)
    avg_completeness = mean(completeness, dim=-1)

    chamfer = avg_accuracy + avg_completeness

    if batched:
        chamfer, accuracy, completeness, indices_pred_gt, indices_gt_pred = unbatchify(
            [chamfer, accuracy, completeness, indices_pred_gt, indices_gt_pred]
        )

    return chamfer, accuracy, completeness, indices_pred_gt, indices_gt_pred


def chamfer_p(prediction: Tensor, groundtruth: Tensor) -> Tensor:
    """Compute the chamfer distance "P".

    The result is computed as:
        - _, cd_pred2gt, cd_gt2pred, _, _ = chamfer(prediction, groundtruth, squared=False)
        - result = (cd_pred2gt.mean(-1) + cd_gt2pred.mean(-1)) / 2.

    Args:
        prediction: The source point cloud(s) with shape ([B,] NUM_POINTS, 3).
        groundtruth: The target point cloud(s) with shape ([B,] NUM_POINTS, 3).

    Returns:
        The computed chamfer distance "P".
    """
    _, cds_pred_gt, cds_gt_pred, _, _ = chamfer(prediction, groundtruth, squared=False)
    cds_pred_gt = torch.mean(cds_pred_gt, dim=-1)
    cds_gt_pred = torch.mean(cds_gt_pred, dim=-1)

    return (cds_gt_pred + cds_pred_gt) / 2


def chamfer_t(prediction: Tensor, groundtruth: Tensor) -> Tensor:
    """Compute the chamfer distance "T".

    The result is computed as:
        - _, cd_pred2gt, cd_gt2pred, _, _ = chamfer(prediction, groundtruth, squared=True)
        - result = cd_pred2gt.mean(-1) + cd_gt2pred.mean(-1).

    Args:
        prediction: The source point cloud(s) with shape ([B,] NUM_POINTS, 3).
        groundtruth: The target point cloud(s) with shape ([B,] NUM_POINTS, 3).

    Returns:
        The computed chamfer distance "T".
    """
    _, cds_pred_gt_sq, cds_gt_pred_sq, _, _ = chamfer(prediction, groundtruth, squared=True)
    cds_pred_gt_sq = torch.mean(cds_pred_gt_sq, dim=-1)
    cds_gt_pred_sq = torch.mean(cds_gt_pred_sq, dim=-1)

    return cds_pred_gt_sq + cds_gt_pred_sq



def f_score(
    predictions: Union[Tensor, None],
    groundtruth: Union[Tensor, None],
    dist_pred_gt: Union[Tensor, None] = None,
    dist_gt_pred: Union[Tensor, None] = None,
    threshold: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute the F1-Score using the treshold as defined in:

    Knapitsch, A., Park, J., Zhou, Q. Y., & Koltun, V. (2017).
    Tanks and temples: Benchmarking large-scale scene reconstruction.
    ACM Transactions on Graphics (ToG), 36(4), 1-13.

    Args:
        predictions: the predicted point cloud(s) with shape ([B,] NUM_POINTS, 3).
        groundtruth: the groundtruth point cloud(s) with shape ([B,] NUM_POINTS, 3).
        dist_pred_gt (optional): per point distances from predictions to groundtruth
            with shape ([B,] NUM_POINTS). Defaults to None.
        dist_gt_pred (optional): per point distances from groundtruth to predictions
            with shape ([B,] NUM_PTS_GT). Defaults to None.
        threshold: the distance treshold to use. Defaults to 0.01.

    Returns:
        A Tuple with:
        - The fscore with shape ([B,]).
        - The precision with shape ([B,]).
        - The recall with shape ([B,]).
    """
    pred_gt = predictions is not None and groundtruth is not None
    distances = dist_pred_gt is not None and dist_gt_pred is not None
    error_msg = (
        "Only one couple between (predictions, grountruths) and "
        "(dist_pred_gt, dist_gt_pred) should be not None"
    )
    assert (not pred_gt and distances) or (pred_gt and not distances), error_msg

    if predictions is not None and groundtruth is not None:
        pred_copy = torch.clone(predictions)
        gt_copy = torch.clone(groundtruth)

        batched, [pred_copy, gt_copy] = batchify([pred_copy, gt_copy], 3)

        dist_precision, _, _ = knn_points(pred_copy, gt_copy)
        dist_precision = dist_precision.squeeze(-1)
        dist_precision = sqrt(dist_precision)

        dist_recall, _, _ = knn_points(gt_copy, pred_copy)
        dist_recall = dist_recall.squeeze(-1)
        dist_recall = sqrt(dist_recall)
    else:
        dist_pred_gt_copy = torch.clone(dist_pred_gt)  # type: ignore
        dist_gt_pred_copy = torch.clone(dist_gt_pred)  # type: ignore

        batched, [dist_precision, dist_recall] = batchify([dist_pred_gt_copy, dist_gt_pred_copy], 2)

    batch_size = len(dist_precision)
    f_scores = zeros(batch_size).to(dist_precision.device)
    recall = zeros(batch_size).to(dist_precision.device)
    precision = zeros(batch_size).to(dist_precision.device)

    num_samples_prec = dist_precision.shape[1]
    num_samples_rec = dist_recall.shape[1]

    precision = sum((dist_precision < threshold), dim=-1) / max(num_samples_prec, 1)
    recall = sum((dist_recall < threshold), dim=-1) / max(num_samples_rec, 1)

    prec_and_rec = recall + precision
    mask = prec_and_rec > 0

    f_scores[mask] = 2 * recall[mask] * precision[mask] / prec_and_rec[mask]

    if batched:
        f_scores, precision, recall = unbatchify([f_scores, precision, recall])

    return f_scores, precision, recall
