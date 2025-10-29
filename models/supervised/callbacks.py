import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader


class DataLoaderRecreationCallback(L.Callback):
    """
    Lightning callback that recreates the training dataloader at each epoch
    with updated sampling weights from the adaptive sampler.

    This ensures that the WeightedRandomSampler reflects the current epoch's
    target distribution d_target(t), allowing the curriculum learning strategy
    to progressively shift from imbalanced to balanced sampling.

    Args:
        train_dataset: Training dataset instance
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        persistent_workers: Whether to keep workers alive between epochs
        pin_memory: Whether to pin memory for faster GPU transfer
        collate_fn: Optional collate function for batching
    """

    def __init__(
        self,
        train_dataset,
        batch_size: int,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = True,
        collate_fn=None
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Called at the start of each training epoch.

        Recreates the training dataloader with updated sampler weights
        based on the current epoch and previous F1 score.
        """
        # Only proceed if adaptive sampling is enabled
        if pl_module.adaptive_sampler is None:
            return

        # Get the current epoch number and F1 score
        current_epoch = trainer.current_epoch
        f1_score = pl_module.val_f1_score

        # Create new weighted sampler for this epoch
        new_sampler = pl_module.adaptive_sampler.get_weighted_sampler(
            epoch=current_epoch,
            f1_score=f1_score
        )

        # Create new dataloader with the updated sampler
        new_train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=new_sampler,  # Use the new sampler
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

        # Replace the trainer's train dataloader
        # Lightning stores the combined loader in the fit loop's data fetcher
        # We need to update both the data source and reset the data fetcher
        try:
            # For Lightning 2.x
            trainer.fit_loop._data_source.instance = new_train_dataloader
            # Reset the data fetcher to use the new dataloader
            if hasattr(trainer.fit_loop, '_data_fetcher'):
                trainer.fit_loop._data_fetcher = None
        except AttributeError:
            # Fallback for different Lightning versions
            try:
                trainer.fit_loop._data_source._train_dataloader_source = new_train_dataloader
            except:
                print("Warning: Could not replace dataloader. Lightning API may have changed.")

        # Log info every 10 epochs
        if current_epoch % 10 == 0:
            info = pl_module.adaptive_sampler.get_distribution_info()
            print(f"\n[Epoch {current_epoch}] Recreated DataLoader with updated sampler")
            print(f"  d_target: FG={info['d_target'][1]:.4f}, BG={info['d_target'][0]:.4f}")
            print(f"  F1 score: {f1_score:.4f}\n")


class AdaptiveSamplingCallback(L.Callback):
    """
    Lightning callback for monitoring and logging adaptive sampling behavior.

    Logs:
    - Target distribution d_target evolution over epochs
    - Loss weights evolution
    - F1 score used for h(t)
    - Sample distribution (foreground vs background ratio)

    Optionally creates plots of distribution evolution.
    """

    def __init__(self, log_dir=None, plot_frequency=10):
        """
        Args:
            log_dir: Directory to save plots (if None, no plots are saved)
            plot_frequency: Save plots every N epochs
        """
        super().__init__()
        self.log_dir = Path(log_dir) if log_dir else None
        self.plot_frequency = plot_frequency

        # Storage for tracking evolution
        self.epochs = []
        self.d_target_fg = []
        self.d_target_bg = []
        self.loss_weight_fg = []
        self.loss_weight_bg = []
        self.f1_scores = []

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Called at the start of each training epoch.

        Logs the current adaptive sampling distribution.
        """
        if pl_module.adaptive_sampler is None:
            return

        # Get current distribution info
        info = pl_module.adaptive_sampler.get_distribution_info()

        # Store for plotting
        self.epochs.append(trainer.current_epoch)
        self.d_target_fg.append(info['d_target'][1])
        self.d_target_bg.append(info['d_target'][0])
        self.loss_weight_fg.append(info['loss_weights'][1])
        self.loss_weight_bg.append(info['loss_weights'][0])
        self.f1_scores.append(info['f1_score'])

        # Log to trainer
        pl_module.log('adaptive/d_target_fg', info['d_target'][1], on_step=False, on_epoch=True)
        pl_module.log('adaptive/d_target_bg', info['d_target'][0], on_step=False, on_epoch=True)
        pl_module.log('adaptive/loss_weight_fg', info['loss_weights'][1], on_step=False, on_epoch=True)
        pl_module.log('adaptive/loss_weight_bg', info['loss_weights'][0], on_step=False, on_epoch=True)
        pl_module.log('adaptive/f1_score', info['f1_score'], on_step=False, on_epoch=True)

        # Print summary
        if trainer.current_epoch % 10 == 0:
            print(f"\n[Epoch {trainer.current_epoch}] Adaptive Sampling:")
            print(f"  d_target: FG={info['d_target'][1]:.4f}, BG={info['d_target'][0]:.4f}")
            print(f"  Loss weights: FG={info['loss_weights'][1]:.4f}, BG={info['loss_weights'][0]:.4f}")
            print(f"  F1 score: {info['f1_score']:.4f}\n")

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each training epoch.

        Optionally creates plots of distribution evolution.
        """
        if self.log_dir is None or trainer.current_epoch % self.plot_frequency != 0:
            return

        if len(self.epochs) < 2:
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Adaptive Sampling Evolution (Epoch {trainer.current_epoch})', fontsize=16)

        # Plot 1: Target Distribution
        ax1 = axes[0, 0]
        ax1.plot(self.epochs, self.d_target_fg, label='Foreground', marker='o', markersize=3)
        ax1.plot(self.epochs, self.d_target_bg, label='Background', marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('d_target')
        ax1.set_title('Target Distribution Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Loss Weights
        ax2 = axes[0, 1]
        ax2.plot(self.epochs, self.loss_weight_fg, label='Foreground', marker='o', markersize=3)
        ax2.plot(self.epochs, self.loss_weight_bg, label='Background', marker='s', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Weight')
        ax2.set_title('Loss Weights Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: F1 Score
        ax3 = axes[1, 0]
        ax3.plot(self.epochs, self.f1_scores, label='F1 Score', marker='o', markersize=3, color='green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Validation F1 Score (for h(t))')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])

        # Plot 4: Ratio of Foreground to Background
        ax4 = axes[1, 1]
        ratios = [fg / (bg + 1e-8) for fg, bg in zip(self.d_target_fg, self.d_target_bg)]
        ax4.plot(self.epochs, ratios, label='FG/BG Ratio', marker='o', markersize=3, color='purple')
        ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Balanced (1:1)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Foreground / Background Ratio')
        ax4.set_title('Sampling Balance Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.log_dir / f'adaptive_sampling_epoch_{trainer.current_epoch:03d}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved adaptive sampling plot to {plot_path}")

    def on_fit_end(self, trainer, pl_module):
        """
        Called when training ends.

        Creates final summary plot.
        """
        if self.log_dir is None or len(self.epochs) < 2:
            return

        # Create comprehensive final plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle('Adaptive Sampling - Final Summary', fontsize=16, fontweight='bold')

        # Plot 1: Target Distribution
        ax1 = axes[0, 0]
        ax1.plot(self.epochs, self.d_target_fg, label='Foreground', marker='o', linewidth=2)
        ax1.plot(self.epochs, self.d_target_bg, label='Background', marker='s', linewidth=2)
        ax1.fill_between(self.epochs, self.d_target_fg, alpha=0.3)
        ax1.fill_between(self.epochs, self.d_target_bg, alpha=0.3)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('d_target', fontsize=12)
        ax1.set_title('Target Distribution Evolution', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Loss Weights
        ax2 = axes[0, 1]
        ax2.plot(self.epochs, self.loss_weight_fg, label='Foreground', marker='o', linewidth=2)
        ax2.plot(self.epochs, self.loss_weight_bg, label='Background', marker='s', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss Weight', fontsize=12)
        ax2.set_title('Loss Weights Evolution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Plot 3: F1 Score
        ax3 = axes[1, 0]
        ax3.plot(self.epochs, self.f1_scores, label='F1 Score', marker='o', linewidth=2, color='green')
        ax3.fill_between(self.epochs, self.f1_scores, alpha=0.3, color='green')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('F1 Score', fontsize=12)
        ax3.set_title('Validation F1 Score (h(t) component)', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.05])

        # Plot 4: Combined view
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()

        # Plot sampling ratio on left axis
        ratios = [fg / (bg + 1e-8) for fg, bg in zip(self.d_target_fg, self.d_target_bg)]
        line1 = ax4.plot(self.epochs, ratios, label='FG/BG Ratio', marker='o', linewidth=2, color='purple')
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Balanced')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('FG/BG Sampling Ratio', fontsize=12, color='purple')
        ax4.tick_params(axis='y', labelcolor='purple')

        # Plot F1 on right axis
        line2 = ax4_twin.plot(self.epochs, self.f1_scores, label='F1 Score', marker='s', linewidth=2, color='green', alpha=0.7)
        ax4_twin.set_ylabel('F1 Score', fontsize=12, color='green')
        ax4_twin.tick_params(axis='y', labelcolor='green')
        ax4_twin.set_ylim([0, 1.05])

        # Combine legends
        lines = line1 + line2 + [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2)]
        labels = ['FG/BG Ratio', 'F1 Score', 'Balanced']
        ax4.legend(lines, labels, loc='upper left', fontsize=10)

        ax4.set_title('Sampling Balance vs Performance', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save final plot
        final_plot_path = self.log_dir / 'adaptive_sampling_final_summary.png'
        plt.savefig(final_plot_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Adaptive Sampling Summary:")
        print(f"  Initial FG/BG ratio: {self.d_target_fg[0] / (self.d_target_bg[0] + 1e-8):.4f}")
        print(f"  Final FG/BG ratio: {self.d_target_fg[-1] / (self.d_target_bg[-1] + 1e-8):.4f}")
        print(f"  Final F1 score: {self.f1_scores[-1]:.4f}")
        print(f"  Summary plot saved to: {final_plot_path}")
        print(f"{'='*60}\n")
