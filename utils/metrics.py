import torch


def compute_f1_components(pred_centers, target_boxes, target_labels, heatmap_size, hit_dist=45.0, num_classes=None, is_binary=True):
    """
    Compute F1 components (TP, FP, FN) by matching predicted centers to target box centers.

    Args:
        pred_centers: List of tensors (one per batch item), each of shape [N_pred, 3] with [x, y, confidence]
        target_boxes: List of tensors (one per batch item), each of shape [N_gt, 4] with YOLO format [x_center, y_center, w, h] (normalized)
        target_labels: List of tensors (one per batch item), each of shape [N_gt] with class labels
        heatmap_size: Tuple (H, W) representing heatmap dimensions for coordinate conversion
        hit_dist: Distance threshold in pixels for matching predictions to ground truth
        num_classes: Number of classes (for multi-class mode)
        is_binary: If True, ignore class labels and do binary detection

    Returns:
        Tuple of (tp, fp, fn) as GPU tensors (scalars)
    """
    batch_size = len(pred_centers)
    heatmap_h, heatmap_w = heatmap_size

    # Get device from predictions
    device = pred_centers[0].device if len(pred_centers) > 0 and pred_centers[0].numel() > 0 else torch.device('cpu')

    # Initialize counters on GPU
    tp = torch.tensor(0, dtype=torch.long, device=device)
    fp = torch.tensor(0, dtype=torch.long, device=device)
    fn = torch.tensor(0, dtype=torch.long, device=device)

    for batch_idx in range(batch_size):
        preds = pred_centers[batch_idx]  # [N_pred, 3] with [x, y, confidence]
        gt_boxes = target_boxes[batch_idx]  # [N_gt, 4] with YOLO format (normalized)
        gt_labels = target_labels[batch_idx] if target_labels is not None else None  # [N_gt]

        # Convert ground truth boxes to pixel coordinates (extract centers)
        if len(gt_boxes) == 0:
            # No ground truth objects - all predictions are false positives
            fp += len(preds)
            continue

        # Extract centers from YOLO boxes and convert to pixel coordinates
        gt_x_center = gt_boxes[:, 0] * heatmap_w  # [N_gt]
        gt_y_center = gt_boxes[:, 1] * heatmap_h  # [N_gt]
        gt_centers = torch.stack([gt_x_center, gt_y_center], dim=1)  # [N_gt, 2]

        if len(preds) == 0:
            # No predictions - all ground truths are false negatives
            fn += len(gt_boxes)
            continue

        # Extract prediction positions
        pred_positions = preds[:, :2]  # [N_pred, 2] with [x, y]

        # For multi-class: we would need class information from predictions
        # Since detect() doesn't return class info, we'll assume binary mode for now
        # or that all detections are foreground class

        # Compute pairwise distances between predictions and ground truth
        # Shape: [N_pred, N_gt]
        distances = torch.cdist(pred_positions, gt_centers, p=2.0)

        # Match predictions to ground truth using greedy assignment
        # A prediction matches a GT if distance < hit_dist
        matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)
        matched_pred = torch.zeros(len(preds), dtype=torch.bool, device=device)

        # Sort predictions by confidence (highest first) for better matching
        pred_confidences = preds[:, 2]
        sorted_pred_indices = torch.argsort(pred_confidences, descending=True)

        for pred_idx in sorted_pred_indices:
            # Find closest unmatched GT
            pred_dists = distances[pred_idx]  # [N_gt]

            # Mask out already matched GTs
            pred_dists = pred_dists.clone()
            pred_dists[matched_gt] = float('inf')

            # Find closest GT
            min_dist, min_gt_idx = torch.min(pred_dists, dim=0)

            if min_dist < hit_dist:
                # Valid match
                matched_pred[pred_idx] = True
                matched_gt[min_gt_idx] = True

        # Count TP, FP, FN
        tp += matched_pred.sum()
        fp += (~matched_pred).sum()
        fn += (~matched_gt).sum()

    return tp, fp, fn


# TODO: Implement the compute_accuracy function
def compute_accuracy(pred, target, hit_dist=45):
    raise NotImplementedError("This is a placeholder for the actual compute_accuracy function.")