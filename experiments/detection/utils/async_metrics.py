"""
Asynchronous metric computation for object detection training.

Computes detection metrics (mAP, F1, precision, recall) on CPU in a background
thread while GPU training continues, eliminating validation dead time.
"""

import threading
import queue
from typing import Dict, List, Optional
import torch
import numpy as np

try:
    from torchmetrics.detection import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not available in async_metrics")


class AsyncMetricComputer:
    """
    Computes detection metrics asynchronously on CPU thread.

    Design pattern: Data staging + background threading
    - Validation step moves tensors to CPU (non-blocking)
    - Validation epoch end submits job to background thread
    - Training epoch starts immediately without waiting
    - Next validation epoch retrieves and logs previous metrics

    Thread safety:
    - Uses queue.Queue (thread-safe) for job/result passing
    - No shared mutable state between threads
    - Proper lifecycle management with graceful shutdown
    """

    def __init__(self):
        self.worker_thread = None
        self.job_queue = queue.Queue(maxsize=2)  # Limit to 2: current + next
        self.results_queue = queue.Queue()
        self.running = False
        self._lock = threading.Lock()

    def start(self):
        """Start background worker thread."""
        with self._lock:
            if self.running:
                return
            self.running = True
            self.worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="MetricComputerThread"
            )
            self.worker_thread.start()
            print("[AsyncMetricComputer] Worker thread started")

    def stop(self, timeout=30.0):
        """
        Stop worker thread gracefully.

        Args:
            timeout: Maximum time to wait for thread termination (seconds)
        """
        with self._lock:
            if not self.running:
                return
            self.running = False

        # Send sentinel to unblock queue
        try:
            self.job_queue.put(None, timeout=1.0)
        except queue.Full:
            pass

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
            if self.worker_thread.is_alive():
                print(f"[AsyncMetricComputer] Warning: Thread did not terminate within {timeout}s")
            else:
                print("[AsyncMetricComputer] Worker thread stopped")

    def submit_job(self, epoch: int, predictions: List[Dict],
                   targets: List[Dict], compute_f1: bool = True):
        """
        Submit metric computation job (non-blocking).

        Args:
            epoch: Current epoch number
            predictions: List of prediction dicts (already on CPU)
                Each dict: {'boxes': Tensor, 'labels': Tensor, 'scores': Tensor}
            targets: List of target dicts (already on CPU)
                Each dict: {'boxes': Tensor, 'labels': Tensor}
            compute_f1: Whether to compute F1 score
        """
        if not self.running:
            print("[AsyncMetricComputer] Warning: Worker not running, call start() first")
            return

        job = {
            'epoch': epoch,
            'predictions': predictions,
            'targets': targets,
            'compute_f1': compute_f1,
        }

        try:
            # Non-blocking put with fallback to blocking
            self.job_queue.put(job, block=False)
            print(f"[AsyncMetricComputer] Submitted job for epoch {epoch}")
        except queue.Full:
            # Queue full - wait for previous job to complete
            print(f"[AsyncMetricComputer] Queue full, waiting to submit epoch {epoch}")
            self.job_queue.put(job, block=True, timeout=10.0)

    def get_results(self, timeout=0.1) -> Optional[Dict]:
        """
        Get computed results if available (non-blocking by default).

        Args:
            timeout: Maximum time to wait for results (seconds)

        Returns:
            Dict with keys: 'epoch', 'map_50', 'map_75', 'map_50_95',
                           'f1', 'precision', 'recall'
            or None if no results ready
        """
        try:
            results = self.results_queue.get(block=True, timeout=timeout)
            print(f"[AsyncMetricComputer] Retrieved results for epoch {results.get('epoch', '?')}")
            return results
        except queue.Empty:
            return None

    def _worker_loop(self):
        """Worker thread main loop - runs metric computation jobs."""
        print("[AsyncMetricComputer] Worker loop started")

        # Initialize metric objects on worker thread (CPU-only)
        if TORCHMETRICS_AVAILABLE:
            map_metric = MeanAveragePrecision(
                box_format='xyxy',
                iou_type='bbox',
                class_metrics=False
            )
        else:
            map_metric = None

        while self.running:
            try:
                job = self.job_queue.get(timeout=1.0)

                if job is None:  # Sentinel for shutdown
                    print("[AsyncMetricComputer] Received shutdown signal")
                    break

                # Compute metrics on CPU
                epoch = job['epoch']
                print(f"[AsyncMetricComputer] Computing metrics for epoch {epoch}...")

                results = self._compute_metrics(
                    job['predictions'],
                    job['targets'],
                    map_metric,
                    compute_f1=job['compute_f1']
                )
                results['epoch'] = epoch

                # Put results
                self.results_queue.put(results)
                print(f"[AsyncMetricComputer] Completed epoch {epoch} - mAP@0.5: {results['map_50']:.4f}")

            except queue.Empty:
                continue
            except Exception as e:
                # Log error but continue worker
                print(f"[AsyncMetricComputer] Error in worker: {e}")
                import traceback
                traceback.print_exc()

        print("[AsyncMetricComputer] Worker loop terminated")

    def _compute_metrics(self, predictions: List[Dict], targets: List[Dict],
                        map_metric, compute_f1: bool) -> Dict:
        """
        Compute all metrics (runs on worker thread, CPU-only).

        Args:
            predictions: List of prediction dicts on CPU
            targets: List of target dicts on CPU
            map_metric: torchmetrics MeanAveragePrecision instance
            compute_f1: Whether to compute F1 score

        Returns:
            Dict with metric values
        """
        # Compute mAP using torchmetrics
        if TORCHMETRICS_AVAILABLE and map_metric is not None:
            map_metric.reset()
            map_metric.update(predictions, targets)
            metric_dict = map_metric.compute()

            map_50 = float(metric_dict.get('map_50', 0.0))
            map_75 = float(metric_dict.get('map_75', 0.0))
            map_50_95 = float(metric_dict.get('map', 0.0))
        else:
            map_50 = map_75 = map_50_95 = 0.0

        # Compute F1 (vectorized version)
        if compute_f1:
            f1, precision, recall = self._compute_f1_vectorized(predictions, targets)
        else:
            f1 = precision = recall = 0.0

        return {
            'map_50': map_50,
            'map_75': map_75,
            'map_50_95': map_50_95,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }

    @staticmethod
    def _compute_f1_vectorized(predictions: List[Dict], targets: List[Dict],
                              iou_threshold: float = 0.5) -> tuple:
        """
        Vectorized F1 computation to eliminate per-image GPU→CPU syncs.

        Key optimizations over original implementation:
        1. Batch IoU computation where possible
        2. Use NumPy for matching logic (CPU-optimized)
        3. Eliminate redundant .item() calls that force GPU→CPU sync
        4. Single aggregation at end instead of per-image syncs

        Args:
            predictions: List of prediction dicts (on CPU)
            targets: List of target dicts (on CPU)
            iou_threshold: IoU threshold for positive match (default: 0.5)

        Returns:
            Tuple of (f1, precision, recall)
        """
        from torchvision.ops import box_iou

        total_tp = total_fp = total_fn = 0

        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']  # Already on CPU
            pred_labels = pred['labels']
            pred_scores = pred['scores']
            gt_boxes = target['boxes']
            gt_labels = target['labels']

            # Handle empty cases
            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue
            elif len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue
            elif len(pred_boxes) == 0:
                total_fn += len(gt_boxes)
                continue

            # Sort predictions by score (descending)
            sorted_idx = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sorted_idx]
            pred_labels = pred_labels[sorted_idx]

            # Compute IoU matrix once [num_pred, num_gt]
            # This is O(N*M) but done only once per image
            iou_matrix = box_iou(pred_boxes, gt_boxes)

            # Convert to numpy for efficient matching
            # NumPy operations are highly optimized for CPU
            iou_np = iou_matrix.cpu().numpy() if iou_matrix.is_cuda else iou_matrix.numpy()
            pred_labels_np = pred_labels.cpu().numpy() if pred_labels.is_cuda else pred_labels.numpy()
            gt_labels_np = gt_labels.cpu().numpy() if gt_labels.is_cuda else gt_labels.numpy()

            gt_matched = np.zeros(len(gt_boxes), dtype=bool)

            # Greedy matching: iterate through predictions in score order
            for p_idx in range(len(pred_boxes)):
                ious = iou_np[p_idx]
                max_iou_idx = ious.argmax()
                max_iou = ious[max_iou_idx]

                # Check if this prediction matches a ground truth
                if max_iou >= iou_threshold and pred_labels_np[p_idx] == gt_labels_np[max_iou_idx]:
                    if not gt_matched[max_iou_idx]:
                        # True positive: first match to this GT
                        total_tp += 1
                        gt_matched[max_iou_idx] = True
                    else:
                        # False positive: duplicate match
                        total_fp += 1
                else:
                    # False positive: no match or wrong class
                    total_fp += 1

            # False negatives: unmatched ground truths
            total_fn += (~gt_matched).sum()

        # Compute metrics
        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1, precision, recall
