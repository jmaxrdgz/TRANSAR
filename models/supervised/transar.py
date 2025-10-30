import torch
import torch.nn.functional as F
import lightning as L

from models.helpers import PeakDetect, DistanceNMS
# from utils.metrics import compute_accuracy
from models.supervised.loss import TRANSARLoss


class TRANSAR(L.LightningModule):
    def __init__(self, config, backbone, head):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'head'])

        self.config = config
        self.backbone = backbone
        self.head = head

        self.loss = TRANSARLoss(alpha=0.05, beta=1.0)

        self.peak_detect = PeakDetect()
        self.dist_nms = DistanceNMS()

        # Adaptive sampling
        self.adaptive_sampler = None
        self.val_f1_score = 0.0

        # Metrics for F1 computation
        self.val_tp = 0
        self.val_fp = 0
        self.val_fn = 0


    def forward(self, x):
        features = self.backbone(x)
        
        # Choose last feature layer
        if isinstance(features, (list, tuple)):
            features = features[-1] 
        if features.ndim == 4 and features.shape[-1] == self.head.in_channels:
            features = features.permute(0, 3, 1, 2)  # B,H,W,C → B,C,H,W

        heatmap = self.head(features)
        return heatmap
    

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            heatmap = self.forward(x)
            pred = self.detect(heatmap)
        return pred
    

    def detect(self, heatmap_logits):
        """
        Detect objects from heatmap logits.

        Args:
            heatmap_logits: Model output [B, C, H, W] (logits, before sigmoid)

        Returns:
            Detected peaks after NMS
        """
        with torch.no_grad():
            # Convert logits to probabilities
            heatmap = torch.sigmoid(heatmap_logits)

            # Threshold heatmap
            mask = (heatmap >= self.config.MODEL.HEAT_THS).float()
            heatmap = heatmap * mask

            # Peak detection
            peaks = self.peak_detect(heatmap)

            # NMS
            pred = self.dist_nms(peaks)

        return pred
    

    def training_step(self, batch, batch_idx):
        '''
        Args:
        batch: Dictionary from yolo_collate_fn containing:
               - 'image': Tensor [B, C, H, W]
               - 'boxes': List of Tensors, each [N_i, 4]
               - 'labels': List of Tensors, each [N_i]
        '''
        x = batch['image']  # [B, C, H, W]
        boxes = batch['boxes']  # List of [N_i, 4] tensors
        # labels = batch['labels']  # List of [N_i] tensors (if using multi-class)
        
        # Forward pass
        heatmap_pred = self.forward(x)  # [B, num_classes, H', W']
        
        # Convert YOLO boxes to target heatmap
        # For single-class detection:
        target_heatmap = self._yolo_to_heatmap(
            boxes,
            size=heatmap_pred.shape[-2:],  # Match output size
            sigma=2.0
        )

        # Or for multi-class detection:
        # target_heatmap = self._yolo_to_heatmap_multiclass(
        #     boxes,
        #     labels,
        #     size=heatmap_pred.shape[-2:],
        #     sigma=2.0,
        #     num_classes=self.num_classes
        # )

        # Ensure target is on same device as predictions (Lightning moves model to GPU)
        target_heatmap = target_heatmap.to(heatmap_pred.device)

        assert heatmap_pred.shape == target_heatmap.shape, \
            f"Heatmap shape mismatch: {heatmap_pred.shape} vs {target_heatmap.shape}"

        # Compute loss
        loss = self.loss(heatmap_pred, target_heatmap)
        
        return loss
    

    def validation_step(self, batch, batch_idx):
        x = batch['image']  # [B, C, H, W]
        boxes = batch['boxes']  # List of [N_i, 4] tensors
        # labels = batch['labels']  # List of [N_i] tensors (if using multi-class)

        # Forward pass
        heatmap_pred = self.forward(x)  # [B, num_classes, H', W']

        # Convert YOLO boxes to target heatmap
        # For single-class detection:
        target_heatmap = self._yolo_to_heatmap(
            boxes,
            size=heatmap_pred.shape[-2:],  # Match output size
            sigma=2.0
        )

        # Or for multi-class detection:
        # target_heatmap = self._yolo_to_heatmap_multiclass(
        #     boxes,
        #     labels,
        #     size=heatmap_pred.shape[-2:],
        #     sigma=2.0,
        #     num_classes=self.num_classes
        # )

        # Ensure target is on same device as predictions (Lightning moves model to GPU)
        target_heatmap = target_heatmap.to(heatmap_pred.device)

        assert heatmap_pred.shape == target_heatmap.shape, \
            f"Heatmap shape mismatch: {heatmap_pred.shape} vs {target_heatmap.shape}"

        # Compute loss
        loss = self.loss(heatmap_pred, target_heatmap)

        # Compute metrics
        pred = self.detect(heatmap_pred)
        # accuracy = compute_accuracy(pred, target_heatmap, hit_dist=self.config.MODEL.HIT_DIST)

        # Accumulate F1 components
        tp, fp, fn = self._compute_f1_components(heatmap_pred, target_heatmap)
        self.val_tp += tp
        self.val_fp += fp
        self.val_fn += fn

        self.log('val_loss', loss, prog_bar=True)
        # self.log('val_accuracy', accuracy, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """
        Lightning hook called at the end of validation epoch.

        Computes F1 score for use in adaptive sampling.
        """
        # Compute F1 score from accumulated metrics
        if self.val_tp + self.val_fp > 0:
            precision = self.val_tp / (self.val_tp + self.val_fp)
        else:
            precision = 0.0

        if self.val_tp + self.val_fn > 0:
            recall = self.val_tp / (self.val_tp + self.val_fn)
        else:
            recall = 0.0

        if precision + recall > 0:
            self.val_f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            self.val_f1_score = 0.0

        # Log F1 components
        self.log('val_f1', self.val_f1_score, prog_bar=True)
        self.log('val_precision', precision, prog_bar=False)
        self.log('val_recall', recall, prog_bar=False, sync_dist=True)

    def _compute_f1_components(self, pred_logits, target, threshold=0.5):
        """
        Compute true positives, false positives, and false negatives.

        Args:
            pred_logits: Predicted heatmap LOGITS [B, C, H, W] (before sigmoid)
            target: Target heatmap [B, C, H, W]
            threshold: Threshold for binarizing predictions (after sigmoid)

        Returns:
            Tuple of (tp, fp, fn) as integers
        """
        # Convert logits to probabilities
        pred = torch.sigmoid(pred_logits)

        # Binarize predictions and targets
        pred_binary = (pred > threshold).float()
        target_binary = (target > 0).float()

        # Compute metrics
        tp = ((pred_binary == 1) & (target_binary == 1)).sum().item()
        fp = ((pred_binary == 1) & (target_binary == 0)).sum().item()
        fn = ((pred_binary == 0) & (target_binary == 1)).sum().item()

        return tp, fp, fn
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.TRAIN.LR,
            weight_decay=1e-4
        )
        return optimizer

    def set_adaptive_sampler(self, sampler):
        """
        Set the adaptive sampler for curriculum-based training.

        Args:
            sampler: AdaptiveSampler instance
        """
        self.adaptive_sampler = sampler

    def on_train_epoch_start(self):
        """
        Lightning hook called at the start of each training epoch.

        Updates adaptive sampling distribution and loss weights based on:
        - Current epoch number
        - F1 score from previous validation
        """
        if self.adaptive_sampler is not None:
            # Compute new target distribution and loss weights
            d_target = self.adaptive_sampler.compute_d_target(
                epoch=self.current_epoch,
                f1_score=self.val_f1_score
            )
            loss_weights = self.adaptive_sampler.compute_loss_weights(d_target)

            # Update loss function with new weights
            self.loss.set_adaptive_weights(loss_weights.to(self.device))

            # Log distribution info
            if self.current_epoch % 10 == 0:  # Log every 10 epochs
                info = self.adaptive_sampler.get_distribution_info()
                self.log('d_target_fg', info['d_target'][1], prog_bar=False)
                self.log('d_target_bg', info['d_target'][0], prog_bar=False)
                self.log('loss_weight_fg', info['loss_weights'][1], prog_bar=False)
                self.log('loss_weight_bg', info['loss_weights'][0], prog_bar=False)

    def on_validation_epoch_start(self):
        """Reset validation metrics at the start of validation."""
        self.val_tp = 0
        self.val_fp = 0
        self.val_fn = 0
    
    
    def _yolo_to_heatmap(self, bboxes, size, sigma=10.0, num_classes=1):
        """Convert YOLO boxes to heatmap representation with Gaussian blobs.

        Args:
            bboxes (List[Tensor]): List of normalized YOLO boxes for each image in batch.
                                Each tensor has shape [N, 4] with (x_center, y_center, w, h).
            size (int or tuple): Size of the output heatmap (height, width).
                                If int, assumes square heatmap.
            sigma (float): Standard deviation of Gaussian kernel in pixels. 
                        Controls the spread of the Gaussian blob.
            num_classes (int): Number of classes for multi-class heatmaps.
                            Use 1 for single-class detection.

        Returns:
            Tensor: Heatmap of shape [B, num_classes, H, W] where B is batch size.
        """
        batch_size = len(bboxes)
        
        # Handle size input
        if isinstance(size, int):
            heatmap_h, heatmap_w = size, size
        else:
            heatmap_h, heatmap_w = size
        
        # Initialize heatmap tensor
        # Try to infer device from bboxes, fallback to model's device
        device = None
        for bbox_list in bboxes:
            if len(bbox_list) > 0:
                device = bbox_list.device
                break
        # If all batches are empty, we'll create on CPU and let caller handle device transfer
        if device is None:
            device = torch.device('cpu')

        heatmap = torch.zeros(
            (batch_size, num_classes, heatmap_h, heatmap_w),
            dtype=torch.float32,
            device=device
        )
        
        # Create 2D Gaussian kernel
        kernel_size = int(6 * sigma + 1)  # 3-sigma rule on each side
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size
        
        gaussian_kernel = self._create_gaussian_kernel_2d(kernel_size, sigma).to(device)
        
        # Process each image in batch
        for batch_idx in range(batch_size):
            boxes = bboxes[batch_idx]  # [N, 4]
            
            if len(boxes) == 0:
                continue  # No objects in this image
            
            # Convert normalized coordinates to pixel coordinates
            x_center = boxes[:, 0] * heatmap_w  # [N]
            y_center = boxes[:, 1] * heatmap_h  # [N]
            
            # For each box, place a point at its center
            for obj_idx in range(len(boxes)):
                x_pix = x_center[obj_idx].item()
                y_pix = y_center[obj_idx].item()
                
                # Convert to integer coordinates
                x_int = int(round(x_pix))
                y_int = int(round(y_pix))
                
                # Skip if out of bounds
                if x_int < 0 or x_int >= heatmap_w or y_int < 0 or y_int >= heatmap_h:
                    continue
                
                # Determine class index (for multi-class, you'd need labels)
                # For single-class detection, always use channel 0
                class_idx = 0
                
                # Place a point (1.0) at object center
                heatmap[batch_idx, class_idx, y_int, x_int] = 1.0
        
        # Apply Gaussian convolution to spread the points into blobs
        # Need to reshape for conv2d: [B*C, 1, H, W]
        heatmap_reshaped = heatmap.view(batch_size * num_classes, 1, heatmap_h, heatmap_w)
        
        # Apply Gaussian kernel
        padding = kernel_size // 2
        heatmap_blurred = F.conv2d(
            heatmap_reshaped,
            gaussian_kernel.unsqueeze(0).unsqueeze(0),  # [1, 1, K, K]
            padding=padding
        )
        
        # Reshape back to [B, C, H, W]
        heatmap_blurred = heatmap_blurred.view(batch_size, num_classes, heatmap_h, heatmap_w)
        
        return heatmap_blurred
    

    def _yolo_to_heatmap_multiclass(self, bboxes, labels, size, sigma=2.0, num_classes=None):
        """Convert YOLO boxes to multi-class heatmap representation with Gaussian blobs.

        Args:
            bboxes (List[Tensor]): List of normalized YOLO boxes for each image in batch.
                                Each tensor has shape [N, 4] with (x_center, y_center, w, h).
            labels (List[Tensor]): List of class labels for each image in batch.
                                Each tensor has shape [N] with integer class IDs.
            size (int or tuple): Size of the output heatmap (height, width).
            sigma (float): Standard deviation of Gaussian kernel in pixels.
            num_classes (int): Number of classes. If None, inferred from labels.

        Returns:
            Tensor: Heatmap of shape [B, num_classes, H, W].
        """
        batch_size = len(bboxes)
        
        # Infer num_classes if not provided
        if num_classes is None:
            max_label = max([lbl.max().item() if len(lbl) > 0 else 0 for lbl in labels])
            num_classes = int(max_label) + 1
        
        # Handle size input
        if isinstance(size, int):
            heatmap_h, heatmap_w = size, size
        else:
            heatmap_h, heatmap_w = size
        
        # Initialize heatmap tensor
        # Try to infer device from bboxes, fallback to model's device
        device = None
        for bbox_list in bboxes:
            if len(bbox_list) > 0:
                device = bbox_list.device
                break
        # If all batches are empty, we'll create on CPU and let caller handle device transfer
        if device is None:
            device = torch.device('cpu')

        heatmap = torch.zeros(
            (batch_size, num_classes, heatmap_h, heatmap_w),
            dtype=torch.float32,
            device=device
        )
        
        # Create 2D Gaussian kernel
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        gaussian_kernel = self._create_gaussian_kernel_2d(kernel_size, sigma).to(device)
        
        # Process each image in batch
        for batch_idx in range(batch_size):
            boxes = bboxes[batch_idx]  # [N, 4]
            lbls = labels[batch_idx]   # [N]
            
            if len(boxes) == 0:
                continue
            
            # Convert normalized coordinates to pixel coordinates
            x_center = boxes[:, 0] * heatmap_w
            y_center = boxes[:, 1] * heatmap_h
            
            # For each box, place a point at its center in the appropriate class channel
            for obj_idx in range(len(boxes)):
                x_pix = x_center[obj_idx].item()
                y_pix = y_center[obj_idx].item()
                class_id = lbls[obj_idx].item()
                
                x_int = int(round(x_pix))
                y_int = int(round(y_pix))
                
                # Skip if out of bounds
                if x_int < 0 or x_int >= heatmap_w or y_int < 0 or y_int >= heatmap_h:
                    continue
                if class_id < 0 or class_id >= num_classes:
                    continue
                
                # Place point at object center in corresponding class channel
                heatmap[batch_idx, class_id, y_int, x_int] = 1.0
        
        # Apply Gaussian convolution to each class channel
        heatmap_reshaped = heatmap.view(batch_size * num_classes, 1, heatmap_h, heatmap_w)
        
        padding = kernel_size // 2
        heatmap_blurred = F.conv2d(
            heatmap_reshaped,
            gaussian_kernel.unsqueeze(0).unsqueeze(0),
            padding=padding
        )
        
        heatmap_blurred = heatmap_blurred.view(batch_size, num_classes, heatmap_h, heatmap_w)
        
        return heatmap_blurred

    
    def _create_gaussian_kernel_2d(self, kernel_size, sigma):
        """Create a 2D Gaussian kernel.

        Args:
            kernel_size (int): Size of the kernel (should be odd).
            sigma (float): Standard deviation of the Gaussian.

        Returns:
            Tensor: 2D Gaussian kernel of shape [kernel_size, kernel_size].
        """
        # Create 1D Gaussian
        x = torch.arange(kernel_size, dtype=torch.float32)
        x = x - kernel_size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        
        # Create 2D Gaussian by outer product
        gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        
        # Normalize
        gauss_2d = gauss_2d / gauss_2d.sum()
        
        return gauss_2d

