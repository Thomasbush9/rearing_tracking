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
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray) -> None:
        """Plot the confusion matrix."""
        pass
    def plot_features(self, features: np.ndarray) -> None:
        """Plot the features."""
        pass
    
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

    