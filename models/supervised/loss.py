import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class TRANSARLoss:
    def __init__(self, alpha=0.05, beta=1.0, multiclass=False):
        self.alpha = alpha
        self.beta = beta
        self.multiclass = multiclass

        # Note: We'll use BCEWithLogitsLoss for numerical stability
        # Model outputs logits, loss applies sigmoid internally
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')

        if multiclass:
            self.DiceLoss = DiceLoss(reduction='none')
        else:
            self.DiceLoss = BinaryDiceLoss(reduction='none')

        # Adaptive weights for class balancing (set via set_adaptive_weights)
        self.adaptive_weights = None

    def set_adaptive_weights(self, weights):
        """
        Set adaptive class weights for current epoch.

        Args:
            weights: Tensor of shape [num_classes] with per-class weights
        """
        self.adaptive_weights = weights

    def __call__(self, pred, target):
        """
        Compute weighted BCE + Dice loss as per TRANSAR paper Eq. 1.

        L(y, ŷ) = α·BCE(y, ŷ) + β·Dice(y, ŷ)

        For adaptive sampling (Section 2.2), this is further weighted:
        L_AS^t(y, ŷ) = w(t) · L(y, ŷ)

        Args:
            pred: Predicted heatmap LOGITS [B, C, H, W] (before sigmoid)
            target: Target heatmap [B, C, H, W] with values in [0, 1]

        Returns:
            Scalar loss value
        """
        # For Dice loss, we need probabilities (apply sigmoid to logits)
        pred_probs = torch.sigmoid(pred)

        # Compute per-pixel BCE loss (with logits for numerical stability)
        bce_loss = self.BCELoss(pred, target)  # [B, C, H, W]

        # Compute Dice loss (using probabilities)
        dice_loss = self.DiceLoss(pred_probs, target)  # [B] or scalar

        # Apply adaptive weights if available
        if self.adaptive_weights is not None:
            # Compute per-pixel weights based on target class
            pixel_weights = self._compute_pixel_weights(target, self.adaptive_weights)

            # Apply weights to BCE loss
            bce_loss = bce_loss * pixel_weights

            # For Dice loss, we need to weight per-sample
            # Dice is already reduced per batch, so we compute sample weights
            if dice_loss.numel() > 1:  # If reduction='none', shape is [B]
                sample_weights = self._compute_sample_weights(target, self.adaptive_weights)
                dice_loss = dice_loss * sample_weights

        # Reduce to scalar
        bce_loss = bce_loss.mean()

        # Dice loss reduction
        if isinstance(dice_loss, torch.Tensor) and dice_loss.numel() > 1:
            dice_loss = dice_loss.mean()

        # Combine losses (Equation 1 from paper)
        loss = self.alpha * bce_loss + self.beta * dice_loss

        return loss

    def _compute_sample_weights(self, target, class_weights):
        """
        Compute per-sample weights for Dice loss.

        For each sample in batch, compute average class weight based on
        the proportion of foreground vs background.

        Args:
            target: Target heatmap [B, C, H, W]
            class_weights: Tensor [num_classes] with weights for each class

        Returns:
            Tensor [B] with per-sample weights
        """
        B = target.shape[0]
        sample_weights = torch.ones(B, device=target.device)

        # Ensure class_weights is on the same device as target
        class_weights = class_weights.to(target.device)

        for b in range(B):
            # Count foreground and background pixels
            fg_pixels = (target[b] > 0).sum().float()
            bg_pixels = (target[b] == 0).sum().float()
            total_pixels = fg_pixels + bg_pixels

            if total_pixels > 0:
                # Weighted average of class weights
                fg_ratio = fg_pixels / total_pixels
                bg_ratio = bg_pixels / total_pixels
                sample_weights[b] = fg_ratio * class_weights[1] + bg_ratio * class_weights[0]

        return sample_weights

    def _compute_pixel_weights(self, target, class_weights):
        """
        Convert class-level weights to pixel-level weights.

        For each pixel, assign weight based on which class it belongs to.
        For binary heatmaps: foreground (>0) vs background (==0).

        Args:
            target: Target heatmap [B, C, H, W]
            class_weights: Tensor [num_classes] with weights for each class

        Returns:
            Tensor [B, C, H, W] with per-pixel weights
        """
        device = target.device
        B, C, H, W = target.shape

        # Ensure class_weights is on the same device as target
        class_weights = class_weights.to(device)

        # Initialize weights
        pixel_weights = torch.ones_like(target, device=device)

        if C == 1:
            # Binary case: single channel heatmap
            # Foreground: target > 0, Background: target == 0
            foreground_mask = target > 0
            background_mask = target == 0

            pixel_weights[foreground_mask] = class_weights[1]  # Foreground weight
            pixel_weights[background_mask] = class_weights[0]  # Background weight

        else:
            # Multi-class case: each channel is a class
            for c in range(C):
                # Pixels with value > 0 in channel c belong to class c
                class_mask = target[:, c:c+1, :, :] > 0
                pixel_weights[:, c:c+1, :, :][class_mask] = class_weights[c]

        return pixel_weights
    
    
def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    # Create result tensor on the same device as input
    result = torch.zeros(shape, device=input.device)
    result = result.scatter_(1, input, 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]