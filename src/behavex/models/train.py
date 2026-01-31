
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import List, Literal, Optional
from behavex.models.data import CustomDataset, GRUTrainingArgs
from behavex.models.gru import GRUModel
from behavex.models.viz import Vizualizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import Tensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from datetime import datetime
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import subprocess
import atexit
import signal
import time
import joblib
import os


class Trainer:
    def __init__(
        self,
        project=None,
        model_name: Literal["gru"] = "gru",
        enable_tensorboard: bool = True,
        save_model: Optional[bool] = None):

        self.project = project
        self.model_name = model_name
        self.enable_tensorboard = enable_tensorboard
        self.training_args = self._build_training_args()

        # Override save_model if explicitly provided
        if save_model is not None:
            self.training_args.save_model = save_model
        self.model = self._build_model()
        self.device = self.training_args.device
        self.model = self.model.to(self.device)
        train_dataset, test_dataset = self._build_dataset()
        self.train_loader = DataLoader(train_dataset, batch_size=self.training_args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.training_args.batch_size, shuffle=False)

        # Set loss function based on task type
        self.task_type = self.training_args.task_type
        if self.task_type == "multiclass":
            # Compute class weights for imbalanced data if enabled
            class_weights = self._compute_class_weights() if self.training_args.use_class_weights else None
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            if class_weights is not None:
                print(f"Using class weights: {class_weights.numpy()}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.training_args.learning_rate, weight_decay=self.training_args.weight_decay)
        self.smoothing = False
        self.smoothing_window = 5
        self.train_losses = []
        self.test_losses = []
        self.test_aucs = []  # For binary: AUC, for multiclass: macro F1
        self.test_f1_micro = []  # For multiclass: micro F1
        self.epochs = []
        self.test_epochs = []
        self.test_every_n_epochs = self.training_args.test_every_n_epochs
        self.viz = Vizualizer()
        
        # Set up model directory structure
        self.model_directory = self.project.project_dir / "models"
        self.model_directory.mkdir(parents=True, exist_ok=True)
        
        # Create runs directory for TensorBoard (only if enabled)
        if self.enable_tensorboard:
            self.runs_dir = self.model_directory / "runs"
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.runs_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            self.writer = None
        
        # TensorBoard process tracking
        self.tensorboard_process = None
        if self.enable_tensorboard:
            atexit.register(self._cleanup_tensorboard)
    
    def _start_tensorboard(self):
        """Start TensorBoard in background process"""
        try:
            # Point TensorBoard to the runs directory
            self.tensorboard_process = subprocess.Popen(
                ['tensorboard', '--logdir', str(self.runs_dir), '--port', '6006'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=None if hasattr(os, 'setsid') else None
            )
            # Give TensorBoard time to start
            time.sleep(3)
            print(f"\n{'='*60}")
            print(f"TensorBoard started at: http://localhost:6006")
            print(f"Log directory: {self.runs_dir}")
            print(f"{'='*60}\n")
        except FileNotFoundError:
            print("Warning: TensorBoard not found. Install with: pip install tensorboard")
        except Exception as e:
            print(f"Warning: Could not start TensorBoard: {e}")
    
    def _cleanup_tensorboard(self):
        """Terminate TensorBoard process"""
        if self.tensorboard_process is not None:
            try:
                self.tensorboard_process.terminate()
                self.tensorboard_process.wait(timeout=5)
                print("\nTensorBoard stopped.")
            except subprocess.TimeoutExpired:
                self.tensorboard_process.kill()
            except Exception as e:
                print(f"Warning: Error stopping TensorBoard: {e}")

    def _build_model(self):
        """Build the model based on the model name"""
        if self.model_name == "gru":
            return GRUModel(input_size=self.training_args.input_size, hidden_size=self.training_args.hidden_size, output_size=self.training_args.output_size)
        else:
            raise ValueError(f"Model name {self.model_name} not supported")

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced multiclass data.

        Returns:
            torch.Tensor: Class weights tensor of shape (num_classes,)
        """
        labels = self.project.train_labels
        num_classes = self.training_args.num_classes

        # Count samples per class
        class_counts = np.bincount(labels.astype(int), minlength=num_classes)

        # Avoid division by zero for classes with no samples
        class_counts = np.maximum(class_counts, 1)

        # Inverse frequency weighting: weight = total_samples / (num_classes * class_count)
        total_samples = len(labels)
        weights = total_samples / (num_classes * class_counts)

        # Normalize so weights sum to num_classes (keeps loss magnitude similar)
        weights = weights / weights.sum() * num_classes

        print(f"Class distribution: {np.bincount(labels.astype(int), minlength=num_classes)}")
        print(f"Computed class weights: {weights}")

        return torch.tensor(weights, dtype=torch.float32).to(self.training_args.device)

    def _build_training_args(self):
        """Extract training config from project and build GRUTrainingArgs"""
        train_cfg = self.project.config.get("model_defaults", {}).get("training", {})
        debug_cfg = self.project.config.get("debug", {})
        annotation_cfg = self.project.config.get("annotation", {})

        # Determine task type and number of classes
        task_type = train_cfg.get("task_type", "binary")
        class_names = []

        if task_type == "multiclass":
            # Get class names from config (behaviors list)
            behaviors = annotation_cfg.get("behaviors", [])
            if not behaviors:
                # Fallback to single behavior
                single_behavior = annotation_cfg.get("behavior", "rearing")
                behaviors = [single_behavior]
            class_names = behaviors
            # num_classes includes background (class 0) + behavior classes
            num_classes = len(behaviors) + 1  # +1 for background
            output_size = num_classes
        else:
            # Binary classification
            num_classes = 2
            output_size = 1

        return GRUTrainingArgs(
            batch_size=train_cfg.get("batch_size", 32),
            input_size=len(self.project.config["data"]["feature_set"]),
            hidden_size=self.project.config["model_defaults"]["training"].get("hidden_size", 64),
            output_size=output_size,
            device=train_cfg.get("device", "mps"),
            learning_rate=train_cfg.get("lr", 1e-3),
            epochs=train_cfg.get("epochs", 20),
            weight_decay=train_cfg.get("weight_decay", 1e-5),  # default
            patience=train_cfg.get("patience", 5),  # default
            verbose=debug_cfg.get("verbose", False),
            save_model=train_cfg.get("save_model", True),  # default
            save_path=str(self.project.project_dir / "models"),  # derive from project
            test_every_n_epochs=train_cfg.get("test_every_n_epochs", 1),
            task_type=task_type,
            num_classes=num_classes,
            class_names=class_names,
            use_class_weights=train_cfg.get("use_class_weights", True),  # Enable by default for multiclass
        )
    def _build_dataset(self):
        """Build the training and test dataset based on the training args"""
        # Reshape windows from (N, T, F) to (N*T, F) for scaling
        original_shape = self.project.train_windows.shape
        windows_2d = self.project.train_windows.reshape(-1, original_shape[-1])

        # Fit scaler on training data only
        self.scaler = StandardScaler()
        windows_scaled_2d = self.scaler.fit_transform(windows_2d)

        # Reshape back to (N, T, F)
        self.project.train_windows = windows_scaled_2d.reshape(original_shape)

        # Don't scale labels - they're class indices or binary (0/1)
        X_train, X_test, y_train, y_test = train_test_split(
            self.project.train_windows,
            self.project.train_labels,
            test_size=0.1,
            random_state=42
        )
        train_dataset = CustomDataset(
            X_windows=X_train,
            y=y_train,
            task_type=self.training_args.task_type
        )
        test_dataset = CustomDataset(
            X_windows=X_test,
            y=y_test,
            task_type=self.training_args.task_type
        )
        return train_dataset, test_dataset
    
    def learning_step(self, X, y):
        """Perform one learning step: forward, backward, optimizer step"""
        X = X.to(self.device)
        y = y.to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(X)
        # Do NOT apply smoothing during training - it breaks gradients
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def train(self):
        """Train the model
        
        Returns:
            float: Best test loss achieved during training
        """
        # Start TensorBoard (only if enabled)
        if self.enable_tensorboard:
            self._start_tensorboard()
        
        best_test_loss = float('inf')
        
        try:
            self.model.train()
            for epoch in range(self.training_args.epochs):
                running_loss = 0.0
                for X, y in self.train_loader:
                    loss = self.learning_step(X, y)
                    if np.isnan(loss):
                        print(f"NaN loss detected at epoch {epoch+1}, stopping training")
                        return best_test_loss if best_test_loss != float('inf') else None
                    running_loss += loss * X.size(0)
                avg_loss = running_loss / len(self.train_loader.dataset)
                self.train_losses.append(avg_loss)
                self.epochs.append(epoch + 1)
                if self.writer is not None:
                    self.writer.add_scalar('train/loss', avg_loss, epoch + 1)
                
                print(f"Epoch {epoch+1}/{self.training_args.epochs} - Train Loss: {avg_loss:.4f}", end="")
                if (epoch + 1) % self.test_every_n_epochs == 0:
                    test_loss, test_metric = self.test()
                    self.test_losses.append(test_loss)
                    self.test_aucs.append(test_metric)  # For multiclass: macro F1, for binary: AUC
                    self.test_epochs.append(epoch + 1)
                    if self.writer is not None:
                        self.writer.add_scalar('test/loss', test_loss, epoch + 1)
                        metric_name = 'test/f1_macro' if self.task_type == 'multiclass' else 'test/auc'
                        self.writer.add_scalar(metric_name, test_metric, epoch + 1)

                        # Log per-class F1 scores for multiclass
                        if self.task_type == 'multiclass' and hasattr(self, 'last_per_class_f1'):
                            class_names = ["background"] + self.training_args.class_names
                            for i, f1_val in enumerate(self.last_per_class_f1):
                                class_name = class_names[i] if i < len(class_names) else f"class_{i}"
                                self.writer.add_scalar(f'test/f1_{class_name}', f1_val, epoch + 1)

                    metric_label = "F1 Macro" if self.task_type == "multiclass" else "AUC"
                    print(f" | Test Loss: {test_loss:.4f} | Test {metric_label}: {test_metric:.4f}")
                    # Track best test loss
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                else:
                    print()
                
                # Run validation every 5 epochs (only if tensorboard enabled)
                if self.enable_tensorboard and (epoch + 1) % 5 == 0:
                    self.validation(epoch+1)
            
            # Plot and log loss history (only if tensorboard enabled)
            if self.enable_tensorboard:
                loss_fig = self.plot_loss_history()
                if self.writer is not None:
                    self.writer.add_figure('loss_history', loss_fig, self.training_args.epochs)
                plt.close(loss_fig)

            # Save confusion matrix for multiclass
            if self.task_type == "multiclass" and hasattr(self, 'last_confusion_matrix'):
                self._save_confusion_matrix()

            if self.training_args.save_model:
                self.save_model()
        finally:
            # Ensure TensorBoard is stopped even if training fails
            if self.enable_tensorboard:
                self._cleanup_tensorboard()
            if self.writer is not None:
                self.writer.close()
        
        return best_test_loss if best_test_loss != float('inf') else (self.test_losses[-1] if self.test_losses else None)
    def plot_loss_history(self):
        plot_dir = self.project.project_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = plot_dir / f"loss_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        return self.viz.plot_loss_history(self.train_losses, self.test_losses, save_path=save_path)

    def _save_confusion_matrix(self):
        """Save confusion matrix as image and CSV at end of training."""
        if not hasattr(self, 'last_confusion_matrix'):
            return

        cm = self.last_confusion_matrix
        class_names = ["background"] + self.training_args.class_names

        # Create output directory
        plot_dir = self.project.project_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save as image
        fig = self.viz.plot_confusion_matrix(
            cm, class_names=class_names,
            save_path=plot_dir / f"confusion_matrix_{timestamp}.png",
            title="Final Confusion Matrix"
        )
        if self.writer is not None:
            self.writer.add_figure('confusion_matrix', fig, self.training_args.epochs)
        plt.close(fig)

        # Save as CSV
        import pandas as pd
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(plot_dir / f"confusion_matrix_{timestamp}.csv")

        # Print classification report
        if hasattr(self, 'last_per_class_f1'):
            print("\n" + "="*50)
            print("Final Classification Report:")
            print("="*50)
            for i, (name, f1_val) in enumerate(zip(class_names, self.last_per_class_f1)):
                print(f"  {name}: F1={f1_val:.4f}")
            print(f"  Macro F1: {self.test_aucs[-1]:.4f}" if self.test_aucs else "")
            print("="*50 + "\n")

    def test(self):
        """Evaluate the model on test set

        Returns:
            tuple: (avg_loss, metric)
                - For binary: metric is AUC
                - For multiclass: metric is macro F1 score
        """
        self.model.eval()
        running_loss = 0.0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                output = self.model(X)

                # Apply smoothing if enabled (only for binary during evaluation)
                if self.task_type == "binary" and getattr(self, "smoothing", False):
                    from scipy.signal import savgol_filter
                    if output.dim() == 2 and output.size(1) == 1:
                        output_np = output.cpu().numpy()
                        window_length = min(getattr(self, "smoothing_window", 5), output_np.shape[0])
                        if window_length % 2 == 0:
                            window_length -= 1
                        if window_length >= 3:
                            output_np = savgol_filter(output_np, window_length, polyorder=2, axis=0)
                        output = torch.from_numpy(output_np).to(output.device).type_as(output)

                loss = self.criterion(output, y)
                running_loss += loss.item() * X.size(0)

                all_outputs.append(output.cpu())
                all_labels.append(y.cpu())

        avg_loss = running_loss / len(self.test_loader.dataset)

        # Concatenate all batches
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        if self.task_type == "multiclass":
            # For multiclass: compute F1 scores
            # Get predicted class indices
            preds = torch.argmax(all_outputs, dim=1).numpy()
            labels = all_labels.numpy()

            # Compute macro F1 (average F1 across all classes)
            macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
            micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)

            # Compute per-class F1 scores
            per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
            self.last_per_class_f1 = per_class_f1  # Store for logging

            # Compute confusion matrix
            self.last_confusion_matrix = confusion_matrix(labels, preds,
                                                          labels=range(self.training_args.num_classes))

            self.model.train()
            return avg_loss, macro_f1
        else:
            # For binary: compute AUC
            try:
                auc = roc_auc_score(all_labels.numpy(), all_outputs.numpy())
            except ValueError:
                # AUC undefined if only one class present
                auc = 0.5
            self.model.train()
            return avg_loss, auc
    
    def save_model(self, model_name: str = None):
        """Save the model with metadata for future loading
        
        Args:
            model_name: Optional custom name, otherwise auto-generated
        """
        # Save models in models/models/ subdirectory
        models_subdir = self.model_directory / "models"
        models_subdir.mkdir(parents=True, exist_ok=True)
        
        # Generate descriptive filename
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{self.model_name}_h{self.training_args.hidden_size}_e{self.training_args.epochs}_{timestamp}"
        
        model_path = models_subdir / f"{model_name}.pth"
        metadata_path = models_subdir / f"{model_name}_metadata.json"
        
        # Save model state dict
        torch.save(self.model.state_dict(), model_path)
        
        # Save metadata for reconstruction
        metadata = {
            "model_name": self.model_name,
            "model_path": str(model_path),
            "timestamp": datetime.now().isoformat(),
            "schema_version": "2.0",  # Version for multiclass support
            "training_args": {
                "input_size": self.training_args.input_size,
                "hidden_size": self.training_args.hidden_size,
                "output_size": self.training_args.output_size,
                "batch_size": self.training_args.batch_size,
                "learning_rate": self.training_args.learning_rate,
                "epochs": self.training_args.epochs,
                "weight_decay": self.training_args.weight_decay,
                "device": self.training_args.device,
                "task_type": self.training_args.task_type,
                "num_classes": self.training_args.num_classes,
            },
            "class_names": self.training_args.class_names,  # List of behavior names
            "feature_set": self.project.config["data"]["feature_set"],
            "window_size": self.project.config["data"]["window_size"],
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_test_loss": self.test_losses[-1] if self.test_losses else None,
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save scaler
        scalers_dir = self.model_directory / "scalers"
        scalers_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = scalers_dir / f"scaler_{model_name}.pkl"
        joblib.dump(self.scaler, scaler_path)
        metadata["scaler_path"] = str(scaler_path)
        
        # Update metadata file with scaler path
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update model registry
        self._update_model_registry(model_name, metadata)
        
        print(f"Model saved: {model_path}")
        print(f"Metadata saved: {metadata_path}")
        print(f"Scaler saved: {scaler_path}")
        return model_path

    def _update_model_registry(self, model_name: str, metadata: dict):
        """Update or create model registry index"""
        registry_path = self.model_directory / "model_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {}
        
        registry[model_name] = {
            "model_path": metadata["model_path"],
            "metadata_path": str(self.model_directory / "models" / f"{model_name}_metadata.json"),
            "timestamp": metadata["timestamp"],
            "hidden_size": metadata["training_args"]["hidden_size"],
            "epochs": metadata["training_args"]["epochs"],
            "final_test_loss": metadata["final_test_loss"],
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2, sort_keys=True)

    def load_model(self, model_identifier: str = None):
        """Load a model by name or path
        
        Args:
            model_identifier: Model name (from registry) or full path to .pth file
                             If None, loads the most recent model
        """
        models_subdir = self.model_directory / "models"
        
        if model_identifier is None:
            # Load most recent model
            model_path = self._get_latest_model()
        elif Path(model_identifier).exists():
            # Direct path provided
            model_path = Path(model_identifier)
        else:
            # Model name from registry
            model_path = models_subdir / f"{model_identifier}.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load metadata if available
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            # Verify architecture matches
            if metadata["training_args"]["input_size"] != self.training_args.input_size:
                raise ValueError(f"Model input_size ({metadata['training_args']['input_size']}) "
                               f"doesn't match current config ({self.training_args.input_size})")
            if metadata["training_args"]["hidden_size"] != self.training_args.hidden_size:
                print(f"Warning: Model hidden_size ({metadata['training_args']['hidden_size']}) "
                      f"differs from current config ({self.training_args.hidden_size})")
                # Rebuild model with correct architecture
                self.training_args.hidden_size = metadata["training_args"]["hidden_size"]
                self.model = self._build_model().to(self.device)
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f"Model loaded: {model_path}")
        return metadata if metadata_path.exists() else None

    def _get_latest_model(self) -> Path:
        """Get the most recently saved model"""
        registry_path = self.model_directory / "model_registry.json"
        if not registry_path.exists():
            raise FileNotFoundError("No models found. Train and save a model first.")
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Sort by timestamp and return most recent
        latest = max(registry.items(), key=lambda x: x[1]["timestamp"])
        return Path(latest[1]["model_path"])

    @staticmethod
    def list_models(project_dir: str) -> dict:
        """List all available models in a project
        
        Args:
            project_dir: Path to project directory
            
        Returns:
            Dictionary of model_name -> metadata
        """
        models_dir = Path(project_dir) / "models"
        registry_path = models_dir / "model_registry.json"
        
        if not registry_path.exists():
            return {}
        
        with open(registry_path, 'r') as f:
            return json.load(f)
    
    def validation(self, epoch: int):
        """Run validation over a temporal consistent timeseries for each session
        Saves plot for each run, session into session folder under validation folder
        """
        for session in self.project.sessions:
            # Skip if no validation data
            if not hasattr(session, 'val_windows') or session.val_windows is None:
                continue

            session_dir = session.session_root / "validation"
            session_dir.mkdir(parents=True, exist_ok=True)
            save_path = session_dir / f"{session.id}_{epoch}_validation.png"

            # Scale validation data using the same scaler as training data
            original_shape = session.val_windows.shape
            windows_2d = session.val_windows.reshape(-1, original_shape[-1])
            windows_scaled_2d = self.scaler.transform(windows_2d)
            val_windows_scaled = windows_scaled_2d.reshape(original_shape)

            # Run model over validation data
            with torch.no_grad():
                validation_data = Tensor(val_windows_scaled).float().to(self.device)
                predictions = self.model(validation_data)

                if self.task_type == "multiclass":
                    # For multiclass: get probabilities via softmax and predicted class
                    probs = torch.softmax(predictions, dim=1).cpu().numpy()
                    pred_classes = np.argmax(probs, axis=1)
                else:
                    # For binary: get probabilities via sigmoid
                    probs = torch.sigmoid(predictions).cpu().numpy().squeeze()

            # Ensure true_labels is 1D
            true_labels = np.array(session.val_labels).flatten()

            # Get actual frame indices (original frame numbers from temporal split)
            if hasattr(session, 'val_frame_indices') and session.val_frame_indices is not None:
                frame_indices = session.val_frame_indices
            elif hasattr(session, 'val_indices') and session.val_indices is not None:
                frame_indices = session.val_indices
            else:
                frame_indices = np.arange(len(true_labels))

            if self.task_type == "multiclass":
                # For multiclass validation
                if probs.shape[0] != len(true_labels):
                    print(f"Warning: Shape mismatch for session {session.id}. Predictions: {probs.shape[0]}, Labels: {len(true_labels)}")
                    continue

                # Print class-wise statistics
                print(f"Session {session.id} - Multiclass Validation: {len(true_labels)} frames")
                unique_true, counts_true = np.unique(true_labels, return_counts=True)
                for cls_idx, count in zip(unique_true, counts_true):
                    cls_idx_int = int(cls_idx)
                    if cls_idx_int == 0:
                        cls_name = "background"
                    elif cls_idx_int - 1 < len(self.training_args.class_names):
                        cls_name = self.training_args.class_names[cls_idx_int - 1]
                    else:
                        cls_name = f"class_{cls_idx_int}"
                    print(f"  Class {cls_idx_int} ({cls_name}): {count} frames")

                # Compute per-class F1
                macro_f1 = f1_score(true_labels, pred_classes, average='macro', zero_division=0)
                print(f"  Macro F1: {macro_f1:.4f}")

                # Plot multiclass validation (plot predicted class vs true class)
                val_fig = self.viz.plot_multiclass_validation(
                    pred_classes, true_labels, probs,
                    class_names=["background"] + self.training_args.class_names,
                    save_path=save_path, epoch=epoch, frame_indices=frame_indices
                )
            else:
                # For binary validation
                if probs.ndim > 1:
                    probs = probs.flatten()

                if len(probs) != len(true_labels):
                    print(f"Warning: Shape mismatch for session {session.id}. Predictions: {probs.shape}, Labels: {true_labels.shape}")
                    continue

                # Debug: print label statistics
                num_positives = np.sum(true_labels)
                print(f"Session {session.id} - Validation: {len(true_labels)} frames, {num_positives} positive labels")

                val_fig = self.viz.plot_validation_predictions(probs, true_labels, save_path=save_path, epoch=epoch, frame_indices=frame_indices)

            if self.writer is not None:
                self.writer.add_figure(f'validation/{session.id}', val_fig, epoch)
            plt.close(val_fig)

        