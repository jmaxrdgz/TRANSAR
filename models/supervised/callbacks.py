import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader


class DataLoaderRecreationCallback(L.Callback):
    """
    Lightning callback that updates the training sampler weights at each epoch.

    This ensures that the WeightedRandomSampler reflects the current epoch's
    target distribution d_target(t), allowing the curriculum learning strategy
    to progressively shift from imbalanced to balanced sampling.

    Instead of recreating the entire dataloader (which breaks Lightning's internal
    state management), this callback updates the sampler's weights in-place.

    Note: This callback is no longer needed with the new approach, but kept for
    backward compatibility. The sampler weight update now happens in the model's
    on_train_epoch_start hook.
    """

    def __init__(self):
        super().__init__()
        print("Warning: DataLoaderRecreationCallback is deprecated. "
              "Sampler updates now happen in model's on_train_epoch_start hook.")

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Called at the start of each training epoch.

        Updates the sampler weights in-place based on the current epoch
        and previous F1 score.
        """
        # Only proceed if adaptive sampling is enabled
        if pl_module.adaptive_sampler is None:
            return

        # Get the current epoch number and F1 score
        current_epoch = trainer.current_epoch
        f1_score = pl_module.val_f1_score

        # Get the sampler from the dataloader
        train_dataloader = trainer.train_dataloader
        if hasattr(train_dataloader, 'sampler'):
            sampler = train_dataloader.sampler
        elif hasattr(train_dataloader, 'batch_sampler'):
            # In case of batch sampler
            sampler = train_dataloader.batch_sampler.sampler
        else:
            print("Warning: Could not find sampler in dataloader")
            return

        # Update sampler weights in-place
        pl_module.adaptive_sampler.update_sampler_weights(
            sampler=sampler,
            epoch=current_epoch,
            f1_score=f1_score
        )

        # Log info every 10 epochs
        if current_epoch % 10 == 0:
            info = pl_module.adaptive_sampler.get_distribution_info()
            print(f"\n[Epoch {current_epoch}] Updated sampler weights in-place")
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
