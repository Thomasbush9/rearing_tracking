import torch
import torch.nn as nn
from pathlib import Path
import json
from typing import Literal, Tuple, Optional
import numpy as np
import joblib

from behavex.models.gru import GRUModel


class Inference:
    def __init__(self, project=None, model_name: Literal["gru"] = "gru"):
        if project is None:
            raise ValueError("Project must be provided")

        self.project = project
        self.model_name = model_name
        self.model = None
        self.metadata = None
        self.scaler = None
        self.task_type = "binary"  # Default, updated when model is loaded
        self.class_names = []  # Class names for multiclass

        # Set up device (use same logic as Trainer)
        train_cfg = project.config.get("model_defaults", {}).get("training", {})
        self.device = train_cfg.get("device", "mps")

        # Model directory structure
        self.model_directory = project.project_dir / "models"

    def predict(self, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Predict probabilities on time series windows

        Args:
            x: Array of shape (TOT_FRAMES, window_size, features)
            batch_size: Batch size for inference

        Returns:
            For binary: Array of shape (TOT_FRAMES,) with probabilities
            For multiclass: Array of shape (TOT_FRAMES, num_classes) with class probabilities
        """
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            # Process in batches
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch).float().to(self.device)
                output = self.model(batch_tensor)

                if self.task_type == "multiclass":
                    # Softmax for multiclass probabilities
                    probs = torch.softmax(output, dim=1).cpu().numpy()
                else:
                    # Sigmoid for binary probabilities
                    probs = torch.sigmoid(output).cpu().numpy().squeeze()

                all_probs.append(probs)

        if self.task_type == "multiclass":
            return np.vstack(all_probs)
        else:
            return np.concatenate(all_probs)

    def predict_classes(self, x: np.ndarray, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class indices and probabilities for multiclass models

        Args:
            x: Array of shape (TOT_FRAMES, window_size, features)
            batch_size: Batch size for inference

        Returns:
            Tuple of:
                - pred_classes: Array of shape (TOT_FRAMES,) with predicted class indices
                - probs: Array of shape (TOT_FRAMES, num_classes) with class probabilities
        """
        probs = self.predict(x, batch_size)

        if self.task_type == "multiclass":
            pred_classes = np.argmax(probs, axis=1)
        else:
            # For binary: threshold at 0.5
            pred_classes = (probs > 0.5).astype(int)
            probs = np.column_stack([1 - probs, probs])  # Convert to 2-class format

        return pred_classes, probs

    def get_class_name(self, class_idx: int) -> str:
        """Get the class name for a given class index

        Args:
            class_idx: Class index (0 = background for multiclass)

        Returns:
            Class name string
        """
        if self.task_type == "multiclass":
            if class_idx == 0:
                return "background"
            elif 0 < class_idx <= len(self.class_names):
                return self.class_names[class_idx - 1]
        else:
            # Binary
            return "positive" if class_idx == 1 else "negative"
        return f"class_{class_idx}"

    @staticmethod
    def smooth_predictions(pred_classes: np.ndarray, min_bout_frames: int = 5) -> np.ndarray:
        """Apply temporal smoothing to remove short spurious predictions.

        Removes behavior bouts shorter than min_bout_frames by replacing them
        with the surrounding class (typically background).

        Args:
            pred_classes: Array of predicted class indices (N,)
            min_bout_frames: Minimum bout duration in frames. Bouts shorter than
                            this will be smoothed out.

        Returns:
            np.ndarray: Smoothed predictions
        """
        if min_bout_frames <= 1:
            return pred_classes

        smoothed = pred_classes.copy()
        n = len(smoothed)

        # Find all bouts (contiguous regions of same class)
        i = 0
        while i < n:
            current_class = smoothed[i]
            # Find end of this bout
            j = i
            while j < n and smoothed[j] == current_class:
                j += 1
            bout_length = j - i

            # If bout is too short and not at boundaries, replace with surrounding class
            if bout_length < min_bout_frames:
                # Determine replacement class (prefer previous class, else next)
                if i > 0:
                    replacement = smoothed[i - 1]
                elif j < n:
                    replacement = smoothed[j]
                else:
                    replacement = current_class  # Can't smooth single-class sequence

                smoothed[i:j] = replacement

            i = j

        return smoothed

    @staticmethod
    def smooth_predictions_median(pred_classes: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply median filter smoothing to predictions.

        Uses a sliding window median filter which preserves edges better than
        simple averaging.

        Args:
            pred_classes: Array of predicted class indices (N,)
            window_size: Size of the median filter window (should be odd)

        Returns:
            np.ndarray: Smoothed predictions
        """
        from scipy.ndimage import median_filter
        if window_size <= 1:
            return pred_classes
        # Ensure odd window size
        if window_size % 2 == 0:
            window_size += 1
        return median_filter(pred_classes, size=window_size, mode='nearest').astype(pred_classes.dtype)

    def predict_smoothed(self, x: np.ndarray, batch_size: int = 64,
                         smoothing_method: str = "min_bout",
                         min_bout_frames: int = 5,
                         median_window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with temporal smoothing applied.

        Args:
            x: Array of shape (TOT_FRAMES, window_size, features)
            batch_size: Batch size for inference
            smoothing_method: "min_bout" or "median"
            min_bout_frames: For min_bout method, minimum bout duration
            median_window: For median method, filter window size

        Returns:
            Tuple of (smoothed_classes, raw_probs)
        """
        pred_classes, probs = self.predict_classes(x, batch_size)

        if smoothing_method == "min_bout":
            smoothed = self.smooth_predictions(pred_classes, min_bout_frames)
        elif smoothing_method == "median":
            smoothed = self.smooth_predictions_median(pred_classes, median_window)
        else:
            smoothed = pred_classes

        return smoothed, probs
    
    def warm_up(self, window_size: int, num_features: int, batch_size: int = 64):
        """Warm up the compiled model with a dummy forward pass
        
        Args:
            window_size: Window size used in the model
            num_features: Number of features
            batch_size: Batch size for inference
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        with torch.no_grad():
            # Create dummy input matching the expected shape (batch_size, window_size, num_features)
            dummy_input = torch.zeros(batch_size, window_size, num_features).float().to(self.device)
            _ = self.model(dummy_input)  # Triggers compilation if using torch.compile
        
        print(f"Model warmed up with shape: ({batch_size}, {window_size}, {num_features})")
    
    def _build_model(self, input_size: int, hidden_size: int, output_size: int = 1):
        """Build the model based on the model name and architecture parameters"""
        if self.model_name == "gru":
            return GRUModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        else:
            raise ValueError(f"Model name {self.model_name} not supported")
    
    def load_model(self, model_identifier: str = None):
        """Load a model by name or path

        Args:
            model_identifier: Model name (from registry) or full path to .pth file
                             If None, loads the most recent model

        Returns:
            dict: Model metadata if available, None otherwise
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
            self.metadata = metadata

            # Build model with architecture from metadata
            training_args = metadata["training_args"]
            self.model = self._build_model(
                input_size=training_args["input_size"],
                hidden_size=training_args["hidden_size"],
                output_size=training_args["output_size"]
            )

            # Set task type and class names from metadata
            self.task_type = training_args.get("task_type", "binary")
            self.class_names = metadata.get("class_names", [])

            if self.task_type == "multiclass":
                print(f"Multiclass model with {training_args.get('num_classes', 2)} classes: "
                      f"{['background'] + self.class_names}")
        else:
            raise FileNotFoundError(f"Metadata not found: {metadata_path}. Cannot determine model architecture.")

        # Load state dict
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load scaler using model name from path
        model_name = model_path.stem
        self.load_scaler(model_name)

        print(f"Model loaded: {model_path}")
        return metadata
    
    def load_scaler(self, model_name: str):
        """Load the scaler associated with a model
        
        Args:
            model_name: Model name (stem of .pth file, without extension)
        """
        # Load scaler
        scalers_dir = self.model_directory / "scalers"
        scaler_path = scalers_dir / f"scaler_{model_name}.pkl"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded: {scaler_path}")
    
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