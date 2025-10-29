import torch
import torch.nn as nn
import torch.nn.functional as F

class PeakDetect(nn.Module):
    """
    Detects local maxima (peaks) in probability heatmaps.
    
    A pixel is considered a peak if it has the maximum value in its local neighborhood
    and is above zero (after thresholding).
    """
    def __init__(self, kernel_size=3):
        """
        Args:
            kernel_size (int): Size of the local neighborhood for peak detection.
                             Default is 3 (3x3 neighborhood).
        """
        super(PeakDetect, self).__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        
    def forward(self, heatmap):
        """
        Detect peaks in the heatmap.
        
        Args:
            heatmap (torch.Tensor): Shape (B, C, H, W) - thresholded probability heatmap
            
        Returns:
            list: List of tensors, one per batch item. Each tensor has shape (N, 3)
                  where N is the number of peaks and columns are [x, y, confidence]
        """
        batch_size, channels, height, width = heatmap.shape
        
        # Apply max pooling to find local maxima
        # If a pixel equals the max in its neighborhood, it's a local maximum
        max_pooled = F.max_pool2d(
            heatmap, 
            kernel_size=self.kernel_size, 
            stride=1, 
            padding=self.pad
        )
        
        # A peak is where the original value equals the max-pooled value
        # and the value is greater than 0 (survived thresholding)
        peak_mask = (heatmap == max_pooled) & (heatmap > 0)
        
        # Extract peaks for each item in the batch
        peaks_list = []
        for b in range(batch_size):
            batch_peaks = []
            for c in range(channels):
                # Get coordinates of peaks for this channel
                y_coords, x_coords = torch.where(peak_mask[b, c])
                
                if len(y_coords) > 0:
                    # Get confidence values at peak locations
                    confidences = heatmap[b, c, y_coords, x_coords]
                    
                    # Stack as [x, y, confidence]
                    channel_peaks = torch.stack([
                        x_coords.float(),
                        y_coords.float(),
                        confidences
                    ], dim=1)
                    
                    batch_peaks.append(channel_peaks)
            
            # Concatenate all peaks for this batch item
            if batch_peaks:
                peaks_list.append(torch.cat(batch_peaks, dim=0))
            else:
                # No peaks found, return empty tensor
                peaks_list.append(torch.empty((0, 3), device=heatmap.device))
        
        return peaks_list


class DistanceNMS(nn.Module):
    """
    Distance-based Non-Maximum Suppression.
    
    Suppresses peaks that are within a certain distance of each other,
    keeping only the peak with the highest confidence.
    """
    def __init__(self, nms_dist=4.0):
        """
        Args:
            nms_dist (float): Distance threshold for NMS. Peaks within this
                            distance will be suppressed (keeping the one with
                            higher confidence).
        """
        super(DistanceNMS, self).__init__()
        self.nms_dist = nms_dist
        
    def forward(self, peaks_list):
        """
        Apply distance-based NMS to detected peaks.
        
        Args:
            peaks_list (list): List of tensors, one per batch item.
                             Each tensor has shape (N, 3) with [x, y, confidence]
            
        Returns:
            list: List of tensors after NMS, same format as input
        """
        nms_peaks_list = []
        
        for peaks in peaks_list:
            if peaks.shape[0] == 0:
                # No peaks to process
                nms_peaks_list.append(peaks)
                continue
            
            # Sort peaks by confidence (descending)
            confidences = peaks[:, 2]
            sorted_indices = torch.argsort(confidences, descending=True)
            sorted_peaks = peaks[sorted_indices]
            
            # Keep track of which peaks to keep
            keep_mask = torch.ones(sorted_peaks.shape[0], dtype=torch.bool, device=peaks.device)
            
            for i in range(sorted_peaks.shape[0]):
                if not keep_mask[i]:
                    continue
                
                # Get current peak position
                curr_pos = sorted_peaks[i, :2]  # [x, y]
                
                # Compute distances to all remaining peaks
                other_positions = sorted_peaks[i+1:, :2]  # [x, y] for all subsequent peaks
                
                if other_positions.shape[0] > 0:
                    # Euclidean distance
                    distances = torch.sqrt(
                        torch.sum((other_positions - curr_pos.unsqueeze(0)) ** 2, dim=1)
                    )
                    
                    # Suppress peaks within nms_dist
                    suppress_indices = distances < self.nms_dist
                    keep_mask[i+1:][suppress_indices] = False
            
            # Keep only non-suppressed peaks
            nms_peaks = sorted_peaks[keep_mask]
            nms_peaks_list.append(nms_peaks)
        
        return nms_peaks_list