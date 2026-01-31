import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list = None,
                                save_path: Path = None, title: str = "Confusion Matrix"):
        """Plot the confusion matrix.

        Args:
            cm: Confusion matrix array (n_classes x n_classes)
            class_names: List of class names for labels
            save_path: Optional path to save figure
            title: Plot title

        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        if class_names is not None:
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=class_names,
                   yticklabels=class_names)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title(title)
        plt.tight_layout()

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_multiclass_validation(self, pred_classes: np.ndarray, true_labels: np.ndarray,
                                    probs: np.ndarray = None, class_names: list = None,
                                    save_path: Path = None, epoch: int = None,
                                    frame_indices: np.ndarray = None):
        """Plot multiclass validation results.

        Args:
            pred_classes: Predicted class indices (N,)
            true_labels: True class indices (N,)
            probs: Optional class probabilities (N, num_classes)
            class_names: List of class names
            save_path: Optional path to save figure
            epoch: Epoch number for title
            frame_indices: Original frame indices for x-axis

        Returns:
            matplotlib.figure.Figure
        """
        from sklearn.metrics import confusion_matrix as compute_cm

        if frame_indices is None:
            frame_indices = np.arange(len(true_labels))

        num_classes = len(class_names) if class_names else int(max(true_labels.max(), pred_classes.max())) + 1

        # Set up figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))

        # Subplot 1: Predicted vs True class over time
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(frame_indices, true_labels, label='True Class', alpha=0.7, linewidth=1.5)
        ax1.plot(frame_indices, pred_classes, label='Predicted Class', alpha=0.7, linewidth=1.5, linestyle='--')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Class Index')
        ax1.set_title('Predicted vs True Class Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        if class_names:
            ax1.set_yticks(range(len(class_names)))
            ax1.set_yticklabels(class_names, fontsize=8)

        # Subplot 2: Confusion Matrix
        ax2 = fig.add_subplot(2, 2, 2)
        cm = compute_cm(true_labels, pred_classes, labels=range(num_classes))
        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.figure.colorbar(im, ax=ax2)
        if class_names:
            ax2.set(xticks=np.arange(num_classes),
                   yticks=np.arange(num_classes),
                   xticklabels=class_names,
                   yticklabels=class_names)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
            plt.setp(ax2.get_yticklabels(), fontsize=8)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > thresh else "black")
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        ax2.set_title('Confusion Matrix')

        # Subplot 3: Per-class accuracy bar chart
        ax3 = fig.add_subplot(2, 2, 3)
        class_acc = []
        for i in range(num_classes):
            mask = true_labels == i
            if mask.sum() > 0:
                class_acc.append((pred_classes[mask] == i).mean())
            else:
                class_acc.append(0.0)
        bars = ax3.bar(range(num_classes), class_acc, color='steelblue', alpha=0.7)
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Per-Class Accuracy')
        ax3.set_ylim(0, 1.0)
        if class_names:
            ax3.set_xticks(range(num_classes))
            ax3.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        for bar, acc in zip(bars, class_acc):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.2f}', ha='center', va='bottom', fontsize=8)

        # Subplot 4: Class probabilities over time (if provided)
        ax4 = fig.add_subplot(2, 2, 4)
        if probs is not None and probs.ndim == 2:
            for i in range(min(probs.shape[1], num_classes)):
                label = class_names[i] if class_names else f'Class {i}'
                ax4.plot(frame_indices, probs[:, i], label=label, alpha=0.6, linewidth=1)
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Probability')
            ax4.set_title('Class Probabilities Over Time')
            ax4.legend(loc='upper right', fontsize=7)
            ax4.grid(True, alpha=0.3)
        else:
            # Plot prediction correctness
            correct = (pred_classes == true_labels).astype(int)
            ax4.plot(frame_indices, correct, alpha=0.7, linewidth=1)
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Correct (1) / Incorrect (0)')
            ax4.set_title('Prediction Correctness Over Time')
            ax4.grid(True, alpha=0.3)

        if epoch is not None:
            fig.suptitle(f'Multiclass Validation - Epoch {epoch}', fontsize=14, y=1.02)

        plt.tight_layout()

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if fig.canvas is not None and hasattr(fig.canvas, 'draw'):
            fig.canvas.draw()

        return fig
    def plot_features(self, features: np.ndarray) -> None:
        """Plot the features."""
        pass
    
    def plot_labels(self, labels: np.ndarray, annotations_df=None, save_path: Path = None, title: str = "Labels"):
        """Plot binary labels with optional event regions
        
        Args:
            labels: Binary labels array (0/1)
            annotations_df: Optional DataFrame with events (start_frame, end_frame)
            save_path: Optional path to save figure
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(labels, linewidth=1, label='Labels')
        
        # Add shaded regions for events if provided
        if annotations_df is not None and len(annotations_df) > 0:
            for idx, event in annotations_df.iterrows():
                ax.axvspan(event['start_frame'], event['end_frame'], 
                          alpha=0.3, color='green', label='Events' if idx == 0 else '')
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Label (0/1)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()  # Add this line
        return fig
    
    def plot_interactive_predictions(self, probs: np.ndarray, save_path: Path = None, 
                                     events_df=None, threshold: float = 0.8):
        """Create interactive HTML plot of predictions with zoom/pan capabilities
        
        Args:
            probs: Probability array for each frame
            save_path: Path to save HTML file (required)
            events_df: Optional DataFrame with events (start_frame, end_frame, duration, label)
            threshold: Threshold line to display
            
        Returns:
            str: Path to saved HTML file
        """
        if save_path is None:
            raise ValueError("save_path is required for interactive plot")
        
        frames = np.arange(len(probs))
        
        # Create figure
        fig = go.Figure()
        
        # Plot probabilities
        fig.add_trace(go.Scatter(
            x=frames,
            y=probs,
            mode='lines',
            name='Probability',
            line=dict(color='blue', width=1),
            hovertemplate='Frame: %{x}<br>Probability: %{y:.3f}<extra></extra>'
        ))
        
        # Add threshold line
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({threshold})",
            annotation_position="right"
        )
        
        # Add event regions if events_df provided
        if events_df is not None and len(events_df) > 0:
            for _, event in events_df.iterrows():
                fig.add_vrect(
                    x0=event['start_frame'],
                    x1=event['end_frame'],
                    fillcolor="green",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    annotation_text=f"Event {event['index']}",
                    annotation_position="top left"
                )
        
        # Update layout
        fig.update_layout(
            title="Prediction Probabilities Over Time",
            xaxis_title="Frame",
            yaxis_title="Probability",
            hovermode='x unified',
            template="plotly_white",
            width=1200,
            height=600,
            xaxis=dict(
                rangeslider=dict(visible=True),  # Range slider for easy navigation
                type="linear"
            )
        )
        
        # Save as HTML
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        print(f"Interactive plot saved: {save_path}")
        
        return str(save_path)

    