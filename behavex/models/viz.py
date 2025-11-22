import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class Vizualizer:
    """Visualizer for the training and validation loss."""
    def __init__(self, save_dir:Path = Path("plots")):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_validation_predictions(self, predicted_labels: np.ndarray, true_labels: np.ndarray,
                                     save_path: Path = None, epoch: int = None, frame_indices: np.ndarray = None):
        """Plot the validation predictions + real labels overlapping and absolute error
        
        Returns:
            matplotlib.figure.Figure: The figure object for TensorBoard logging
        """
        if frame_indices is None:
            frame_indices = np.arange(len(true_labels))
        
        # Set up figure with 2 plots side by side
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Overlapping predictions vs ground truth
        axes[0].plot(frame_indices, true_labels, label='Ground Truth', alpha=0.7, linewidth=2)
        axes[0].plot(frame_indices, predicted_labels, label='Predictions (prob)', alpha=0.7, linewidth=2)
        axes[0].set_title('Predictions vs Ground Truth')
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Probability')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Prediction error
        errors = np.abs(predicted_labels - true_labels)
        axes[1].plot(frame_indices, errors, label='Absolute Error', alpha=0.7, color='orange', linewidth=2)
        axes[1].set_title('Prediction Error')
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('|Pred - True|')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        if epoch is not None:
            plt.suptitle(f'Validation Predictions - Epoch {epoch}', fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Ensure figure is drawn for TensorBoard
        if fig.canvas is not None and hasattr(fig.canvas, 'draw'):
            fig.canvas.draw()
        
        return fig
        

    def plot_loss_history(self, train_losses: list[float], test_losses: list[float], save_path: Path = None):
        """Plot the loss history
        
        Returns:
            matplotlib.figure.Figure: The figure object for TensorBoard logging
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Train Loss', marker='o', alpha=0.7)
        ax.plot(test_losses, label='Test Loss', marker='s', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Test Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Ensure figure is drawn for TensorBoard
        if fig.canvas is not None and hasattr(fig.canvas, 'draw'):
            fig.canvas.draw()
        
        return fig
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray) -> None:
        """Plot the confusion matrix."""
        pass
    def plot_features(self, features: np.ndarray) -> None:
        """Plot the features."""
        pass

