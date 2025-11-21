import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class Vizualizer:
    """Visualizer for the training and validation loss."""
    def __init__(self, save_dir:Path = Path("plots")):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_validation_predictions(self,):
        """Plot the validation predictions + real labels."""
        pass

    def plot_loss_history(self, train_losses: list[float], test_losses: list[float], save:bool = True) -> None:
        """Plot the loss history"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Train Loss', marker='o', alpha=0.7)
        ax.plot(test_losses, label='Test Loss', marker='s', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Test Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            plt.savefig(self.save_dir / "loss_history.png")
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray) -> None:
        """Plot the confusion matrix."""
        pass
    def plot_features(self, features: np.ndarray) -> None:
        """Plot the features."""
        pass

