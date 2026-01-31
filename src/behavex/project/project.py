from pathlib import Path
from typing import List, Optional
from datetime import datetime
import yaml
import pandas as pd
from behavex.project.session import Session
from behavex.models.train import Trainer
from behavex.inference.inference import Inference
from behavex.models.viz import Vizualizer
from behavex.hpo.HPO import HPO
from behavex.utils.validation import validate_session_consistency, validate_predictions
import numpy as np

def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively update dict 'base' with values from 'updates'"""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base

class Project:
    def __init__(
        self,
        project_dir: str,
        project_name: Optional[str] = None,
    ):
        # init the project directory
        self.project_dir = Path(project_dir)
        self.sessions_path = self.project_dir / "sessions.yaml"
        if not self.project_dir.exists():
            self.project_dir.mkdir(parents=True)

        # Determine project name: use provided, load from config, or use directory name
        config_path = self.project_dir / "config.yaml"
        if project_name is None:
            if config_path.exists():
                # Load existing config to get project name
                with open(config_path) as stream:
                    try:
                        existing_config = yaml.safe_load(stream)
                        if existing_config and "project" in existing_config and "name" in existing_config["project"]:
                            self.project_name = existing_config["project"]["name"]
                        else:
                            # Fallback to directory name if name not in config
                            self.project_name = self.project_dir.name
                    except yaml.YAMLError:
                        # Fallback to directory name on error
                        self.project_name = self.project_dir.name
            else:
                # New project: use directory name
                self.project_name = self.project_dir.name
        else:
            self.project_name = project_name

        # load or create config file
        self.load_config()
        self.load_sessions()
        self.viz = Vizualizer()
        # List of sessions to predict (session IDs or Session objects)
        self.sessions_to_predict = []

    def load_config(
        self,
    ):
        """load config file from the project directory"""
        # check exist:
        self.config_path = self.project_dir / "config.yaml"
        if self.config_path.exists():
            with open(self.config_path) as stream:
                try:
                    self.config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    raise ValueError(f"Error loading config file: {exc}")
            return
            # Otherwise → create default config skeleton
        default_config = {
            "project": {
                "name": self.project_name,
                "version": "1.0",
                "description": "",
            },
            "paths": {
                "video_root": "",
                "keypoints_root": "",
            },
            "data": {
                "fps": 60,
                "window_size": 30,
                "feature_set": [],
            },
            "annotation": {
                "behavior": "rearing",
                "annotator": "",
                "output_format": "csv",
            },
            #TODO: add other params for training
            "model_defaults": {
                "model_name": "gru",
                "training": {
                    "epochs": 20,
                    "device": "mps",
                    "batch_size": 32,
                    "hidden_size": 16,
                    "lr": 1e-3,
                    "weight_decay": 1e-5,
                    "patience": 5,
                    "save_model": True,
                    "save_path": str(self.project_dir / "models"),
                    "verbose": False,
                    "test_every_n_epochs": 1,
                },
                "inference": {
                    "smoothing": True,
                    "smoothing_window": 5,
                },
            },
            "debug": {
                "verbose": False,
                "save_intermediate": False,
            },
            "validation": {
                "enable_checks": True,
                "strict_mode": False,
            },
        }

        # Save skeleton to disk
        with open(self.config_path, "w") as f:
            yaml.safe_dump(default_config, f, sort_keys=False)

        # Store in memory
        self.config = default_config

    def load_sessions(
        self,
    ):
        """Load the session yaml, if not found it init it"""
        if self.sessions_path.exists():
            with open(self.sessions_path) as stream:
                data = yaml.safe_load(stream)
            if data is None:
                data = {"sessions": []}
            # validate the structure
            if not isinstance(data, dict):
                raise ValueError("Sessions.yaml must contain a dict at top level")
            if "sessions" not in data:
                data["sessions"] = []
            if not isinstance(data["sessions"], list):
                raise ValueError("'sessions' must be a list inside sessions.yaml.")

            # Store metadata + create Session objects
            self.sessions_data = data["sessions"]
            self.sessions = [
                Session(metadata=s, project=self) for s in self.sessions_data
            ]

            # Check for and merge duplicate paths (cleanup existing duplicates)
            self._merge_duplicate_sessions()
            if len(self.sessions_data) != len(data["sessions"]):
                # Some duplicates were merged, save the cleaned up sessions
                self.save_sessions()

            return
        default_data = {"sessions": []}
        with open(self.sessions_path, "w") as stream:
            yaml.safe_dump(default_data, stream, sort_keys=False)
        self.sessions_data = []
        self.sessions = []

    def _merge_sessions(self, existing_session_metadata: dict, new_views: dict, new_annotation_view: str = None):
        """
        Merge a new session's views into an existing session.

        Args:
            existing_session_metadata: The existing session's metadata dict (will be modified in place)
            new_views: New views dictionary to merge in
            new_annotation_view: Optional new annotation_view to use if not set in existing
        """
        # Merge views (union - new views override existing if keys conflict)
        existing_session_metadata["views"].update(new_views)

        # Set annotation_view if not already set or if provided
        if existing_session_metadata.get("annotation_view") is None and new_annotation_view is not None:
            existing_session_metadata["annotation_view"] = new_annotation_view
        elif existing_session_metadata.get("annotation_view") not in existing_session_metadata["views"]:
            # If current annotation_view is invalid, use new one or first available
            if new_annotation_view and new_annotation_view in existing_session_metadata["views"]:
                existing_session_metadata["annotation_view"] = new_annotation_view
            elif existing_session_metadata["views"]:
                # Prefer "bottom" view if available, otherwise use first view
                if "bottom" in existing_session_metadata["views"]:
                    existing_session_metadata["annotation_view"] = "bottom"
                else:
                    existing_session_metadata["annotation_view"] = list(existing_session_metadata["views"].keys())[0]

    def _merge_duplicate_sessions(self):
        """
        Detect and merge sessions with duplicate paths.
        Keeps the first session's ID and merges views from duplicates.
        """
        seen_paths = {}
        sessions_to_remove = []

        for idx, session_metadata in enumerate(self.sessions_data):
            session_path = Path(session_metadata.get("video_dir", "")).expanduser().resolve()

            if not session_path or not session_path.exists():
                continue

            # Check if we've seen this path before
            path_key = str(session_path)
            if path_key in seen_paths:
                # Duplicate found - merge with existing
                existing_idx = seen_paths[path_key]
                existing_metadata = self.sessions_data[existing_idx]
                new_views = session_metadata.get("views", {})
                new_annotation_view = session_metadata.get("annotation_view")

                print(f"Warning: Found duplicate session path '{session_path}'. "
                      f"Merging session '{session_metadata.get('id')}' into '{existing_metadata.get('id')}'.")

                self._merge_sessions(existing_metadata, new_views, new_annotation_view)
                sessions_to_remove.append(idx)
            else:
                seen_paths[path_key] = idx

        # Remove duplicates (in reverse order to maintain indices)
        for idx in sorted(sessions_to_remove, reverse=True):
            removed_session = self.sessions_data.pop(idx)
            # Also remove from sessions list
            self.sessions = [s for s in self.sessions if s.id != removed_session.get("id")]

        if sessions_to_remove:
            print(f"Merged {len(sessions_to_remove)} duplicate session(s).")

    def add_session(
        self, session_dir: Path, session_id: str = None, annotation_view: str = None
    ):
        session_dir = Path(session_dir).expanduser().resolve()

        if not session_dir.exists():
            raise ValueError(f"Session directory does not exist: {session_dir}")

        # Check for duplicate path first
        for existing_session in self.sessions:
            existing_path = Path(existing_session.video_dir).expanduser().resolve()
            if session_dir == existing_path:
                # Duplicate path found - merge instead of creating new
                print(f"Session with path '{session_dir}' already exists (ID: '{existing_session.id}'). Merging views.")
                new_views = {}
                video_files = list(session_dir.glob("*.mp4"))
                for f in video_files:
                    fname = f.stem
                    raw_suffix = fname.split("_")[-1]
                    if raw_suffix.startswith("mirror-"):
                        view_name = raw_suffix.replace("mirror-", "")
                    else:
                        view_name = raw_suffix
                    new_views[view_name] = str(f.resolve())

                if annotation_view is None:
                    # Prefer "bottom" view if available, otherwise use first view
                    if "bottom" in new_views:
                        annotation_view = "bottom"
                    else:
                        annotation_view = list(new_views.keys())[0] if new_views else None

                # Merge into existing session
                self._merge_sessions(existing_session.metadata, new_views, annotation_view)

                # Recreate Session object to reflect updated metadata
                existing_idx = next(i for i, s in enumerate(self.sessions) if s.id == existing_session.id)
                self.sessions[existing_idx] = Session(metadata=existing_session.metadata, project=self)

                self.save_sessions()
                print(f"Updated session {existing_session.id} with merged views.")
                return self.sessions[existing_idx]

        # No duplicate found - create new session
        # check id
        if session_id is None:
            existing_ids = {s["id"] for s in self.sessions_data}
            idx = 1
            while f"session_{idx:03d}" in existing_ids:
                idx += 1
            session_id = f"session_{idx:03d}"

        video_files = list(session_dir.glob("*.mp4"))
        if len(video_files) == 0:
            raise ValueError(f"No video files found in {session_dir}")

        # parsing video paths:
        views = {}
        for f in video_files:
            fname = f.stem  # no extension

            # Example fname:
            # multicam_video_2025-05-07T12_16_20_mirror-bottom

            # Extract the suffix after the timestamp:
            # split by "_" and take the last part
            raw_suffix = fname.split("_")[-1]

            # Normalize to canonical view names
            if raw_suffix.startswith("mirror-"):
                view_name = raw_suffix.replace("mirror-", "")
            else:
                view_name = raw_suffix  # e.g. "central"

            views[view_name] = str(f.resolve())

        if annotation_view is None:
            # Prefer "bottom" view if available, otherwise use first view
            if "bottom" in views:
                annotation_view = "bottom"
            else:
                annotation_view = list(views.keys())[0]
        if annotation_view not in views:
            raise ValueError(f"annotation_view {annotation_view} is not in views")
        metadata = {
            "id": session_id,
            "video_dir": str(session_dir),
            "views": views,
            "annotation_view": annotation_view,
        }
        self.sessions_data.append(metadata)
        new_sess = Session(metadata=metadata, project=self)
        self.sessions.append(new_sess)

        self.save_sessions()

        print(f"Added session {session_id} with {len(views)} views.")
        return new_sess

    def save_sessions(self):
        """Write current Session registry to sessions.yaml"""
        data = {"sessions": self.sessions_data}
        with open(self.sessions_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def set_config_value(self, key_path: str, value, save: bool = True):
        """Changes one value in the config file:"""
        if not hasattr(self, "config") or self.config is None:
            raise RuntimeError("Config not loaded. call load_config() first.")
        keys = key_path.split(".")
        cfg = self.config
        for key in keys[:-1]:
            if key not in cfg or not isinstance(cfg[key], dict):
                cfg[key] = {}
            cfg = cfg[key]
        cfg[keys[-1]] = value

        if save:
            with open(self.config_path, "w") as stream:
                yaml.safe_dump(self.config, stream, sort_keys=True)

    def edit_config(self, updates: dict, save: bool = True):
        """Recursively update the existing config with 'updates' dict.
        Only specified keys are changed"""
        if not hasattr(self, "config") or self.config is None:
            raise RuntimeError("Config not loaded. call load_config first")
        _deep_update(self.config, updates)

        if save:
            with open(self.config_path, "w") as stream:
                yaml.safe_dump(self.config, stream, sort_keys=True)

    def show_config(self):
        print(yaml.safe_dump(self.config, sort_keys=True))

    def get_session(
        self,
    ):
        pass

    def import_features_for_session(self, session_id: str, features_path: str | Path, video_frame_count: int = None):
        """
        Import features for a specific session.

        Args:
            session_id: ID of the session (e.g., "session_001")
            features_path: Path to features CSV or Excel file (string or Path)
            video_frame_count: Optional frame count for validation

        Returns:
            Path to imported features file

        Example:
            project.import_features_for_session(
                "session_001",
                "/path/to/features.xlsx",
                video_frame_count=3600
            )
        """
        # Find session
        session = None
        for s in self.sessions:
            if s.id == session_id:
                session = s
                break

        if session is None:
            raise ValueError(f"Session {session_id} not found in project")

        # Import features (handle both string and Path)
        features_path_obj = Path(features_path).expanduser().resolve()
        return session.import_features(features_path_obj, video_frame_count)

    def get_feature_set(self):
        """Get the feature set from the config"""
        return self.config["data"]["feature_set"]

    def select_features(self):
        """Loads the features from the features dataset for each session.
        It iterates through the sessions and loads the features from the features dataset for each session.
        It saves the featuers selected as np.ndarray in the session.features attribute.
        """
        featues_set = self.get_feature_set()
        for session in self.sessions:
            session.select_features(featues_set)


    def merge_and_generate_training_data(self):
        """Merge all the session training data into a single training dataset"""
        # Loop across sessions and merge training labels (1d) and windows (3d array).
        train_labels = []
        train_windows = []
        for session in self.sessions:
            if not hasattr(session, 'train_windows') or session.train_windows is None:
                raise ValueError(f"Session {session.id} training data not split. Call split_data() first.")
            train_labels.extend(session.train_labels)
            train_windows.extend(session.train_windows)
        # Convert labels to a numpy array (TOT_FRAMES,)
        self.train_labels = np.array(train_labels)
        self.train_windows = np.stack(train_windows, axis=0)
        # Ensure labels are 1-dimensional (just 0 and 1)
        if self.train_labels.shape[0] != self.train_windows.shape[0]:
            raise ValueError("Number of labels and windows do not match")
        if self.train_labels.ndim != 1:
            raise ValueError("Labels must be 1-dimensional (array of 0s and 1s)")
        # Check number of features matches feature set length
        num_features = len(self.config["data"]["feature_set"])
        if self.train_windows.shape[2] != num_features:
            raise ValueError("Number of features does not match the feature set")
    def process_sessions_for_training(self):
        """
        Run select_features(), build_windows(), build_labels(), split_data(),
        subsample_session() for each session, then merge training data.

        Supports both binary and multiclass training modes:
        - Binary: Uses build_labels() with optional behavior_filter
        - Multiclass: Uses build_multiclass_labels() with class_names from config

        Ensures labels are built before attempting split_data(), per error in context.
        """
        # Always select features first (across all sessions)
        self.select_features()

        # Determine task type from config
        train_cfg = self.config.get("model_defaults", {}).get("training", {})
        task_type = train_cfg.get("task_type", "binary")
        annotation_cfg = self.config.get("annotation", {})

        # Get behavior filter for binary mode
        behavior_filter = annotation_cfg.get("behavior_filter", None)
        # Fallback to single behavior field for backward compatibility
        if behavior_filter is None:
            behavior_filter = annotation_cfg.get("behavior", None)

        # Get class names for multiclass mode
        class_names = annotation_cfg.get("behaviors", [])
        if not class_names:
            # Fallback to single behavior
            single_behavior = annotation_cfg.get("behavior", "rearing")
            class_names = [single_behavior]

        # Then process each session stepwise
        for session in self.sessions:
            session.load_annotations()
            session.build_windows(self.config["data"]["window_size"])

            if task_type == "multiclass":
                # Multiclass: build labels with class indices
                session.build_multiclass_labels(class_names=class_names)
            else:
                # Binary: build binary labels (0/1)
                session.build_labels(behavior_filter=behavior_filter)

            # Ensure labels exist before split_data
            if not hasattr(session, "labels") or session.labels is None:
                raise ValueError(f"Labels not loaded for session {session.id}. Call build_labels() first.")
            session.split_data()
            session.subsample_session()
        self.merge_and_generate_training_data()

    def train(self, hpo: bool = False, n_trials: int = 10):
        """Launch model training"""
        #TODO: write safety checks before calling trainer:
        if hpo:
            hpo = HPO(project=self)
            hpo.optimize(n_trials=n_trials)
        else:
            trainer = Trainer(project=self)
            trainer.train()

    def tune_window_size(self, window_sizes: list[int]):
        """Tune the window size for the best performance"""
        from datetime import datetime
        from torch.utils.tensorboard import SummaryWriter

        results = {}
        runs_dir = self.project_dir / "models" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        for window_size in window_sizes:
            self.set_config_value("data.window_size", window_size)
            self.process_sessions_for_training()

            trainer = Trainer(project=self, enable_tensorboard=False)
            trainer.writer = SummaryWriter(log_dir=runs_dir / f"window_{window_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            trainer.enable_tensorboard = True
            trainer._start_tensorboard = lambda: None
            trainer._cleanup_tensorboard = lambda: None

            trainer.train()

            best_auc = max(trainer.test_aucs) if trainer.test_aucs else None
            results[window_size] = {
                'best_auc': best_auc,
                'final_auc': trainer.test_aucs[-1] if trainer.test_aucs else None,
                'best_test_loss': min(trainer.test_losses) if trainer.test_losses else None,
            }
            print(f"Window size {window_size}: Best AUC: {best_auc:.4f}, Final AUC: {results[window_size]['final_auc']:.4f}")

        best_window = max(results.keys(), key=lambda w: results[w]['best_auc'] or 0)
        print(f"\nBest window size: {best_window} with AUC: {results[best_window]['best_auc']:.4f}")
        print(f"\nView results: tensorboard --logdir {runs_dir}")

        return results
    def _prepare_session_for_prediction(self, session):
        """Prepare session data for prediction: load features, build windows, scale data

        Args:
            session: Session object to prepare

        Returns:
            np.ndarray: Scaled windows ready for inference, shape (TOT_FRAMES, window_size, features)
        """
        # Load/select features if not already done
        if session.features is None:
            session.select_features(self.config["data"]["feature_set"])

        # Build windows
        window_size = self.config["data"]["window_size"]
        if not hasattr(session, 'windows') or session.windows is None:
            session.build_windows(window_size)

        # Scale windows using the loaded scaler (must be called after Inference.load_model)
        # Reshape windows from (N, T, F) to (N*T, F) for scaling
        original_shape = session.windows.shape
        windows_2d = session.windows.reshape(-1, original_shape[-1])
        windows_scaled_2d = self.inference.scaler.transform(windows_2d)
        windows_scaled = windows_scaled_2d.reshape(original_shape)

        return windows_scaled
    #TODO: make features consistent between training and prediction
    def _probabilities_to_events(self, probs: np.ndarray, threshold: float = 0.8,
                                  class_names: list = None) -> pd.DataFrame:
        """Convert probability array to events DataFrame

        Args:
            probs: For binary: (N,) array of probabilities
                   For multiclass: (N, K) array of class probabilities
            threshold: Threshold for binary classification (ignored for multiclass)
            class_names: List of class names for multiclass (index 0 = background)

        Returns:
            pd.DataFrame: DataFrame with columns: index, start_frame, end_frame, duration, label
        """
        if probs.ndim == 2:
            # Multiclass: get predicted class for each frame
            pred_classes = np.argmax(probs, axis=1)
            return self._classes_to_events(pred_classes, class_names)
        else:
            # Binary classification
            binary = (probs >= threshold).astype(int)

            # Find contiguous regions
            events = []
            in_event = False
            start_frame = None
            default_label = self.config.get("annotation", {}).get("behavior", "rearing")

            for i, pred in enumerate(binary):
                if pred == 1 and not in_event:
                    start_frame = i
                    in_event = True
                elif pred == 0 and in_event:
                    end_frame = i - 1
                    duration = end_frame - start_frame + 1
                    events.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'duration': duration,
                        'label': default_label
                    })
                    in_event = False

            # Handle event that extends to end
            if in_event:
                end_frame = len(binary) - 1
                duration = end_frame - start_frame + 1
                events.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'duration': duration,
                    'label': default_label
                })

            df = pd.DataFrame(events)
            if len(df) > 0:
                df.insert(0, 'index', range(1, len(df) + 1))
            else:
                df = pd.DataFrame(columns=['index', 'start_frame', 'end_frame', 'duration', 'label'])

            return df

    def _classes_to_events(self, pred_classes: np.ndarray, class_names: list = None) -> pd.DataFrame:
        """Convert predicted class indices to events DataFrame for multiclass

        Args:
            pred_classes: (N,) array of predicted class indices (0 = background)
            class_names: List of class names where index 0 = background

        Returns:
            pd.DataFrame: DataFrame with columns: index, start_frame, end_frame, duration, label
        """
        if class_names is None:
            # Default class names
            behaviors = self.config.get("annotation", {}).get("behaviors", ["behavior"])
            class_names = ["background"] + behaviors

        events = []
        current_class = 0  # Start with background
        start_frame = 0

        for i, pred_class in enumerate(pred_classes):
            if pred_class != current_class:
                # End previous event (if it wasn't background)
                if current_class != 0:
                    end_frame = i - 1
                    duration = end_frame - start_frame + 1
                    label = class_names[current_class] if current_class < len(class_names) else f"class_{current_class}"
                    events.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'duration': duration,
                        'label': label
                    })
                # Start new event
                current_class = pred_class
                start_frame = i

        # Handle final event
        if current_class != 0:
            end_frame = len(pred_classes) - 1
            duration = end_frame - start_frame + 1
            label = class_names[current_class] if current_class < len(class_names) else f"class_{current_class}"
            events.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration': duration,
                'label': label
            })

        df = pd.DataFrame(events)
        if len(df) > 0:
            df.insert(0, 'index', range(1, len(df) + 1))
        else:
            df = pd.DataFrame(columns=['index', 'start_frame', 'end_frame', 'duration', 'label'])

        return df

    def predict_session(self, session, model_identifier=None, batch_size=64, threshold=0.5, save=True):
        """Predict on a single session

        Args:
            session: Session object or session ID string
            model_identifier: Model name or path (None for latest)
            batch_size: Batch size for inference
            threshold: Probability threshold for binary classification
            save: Whether to save predictions to CSV

        Returns:
            np.ndarray: Probabilities array for each frame
        """
        # Convert session ID to Session object if needed
        if isinstance(session, str):
            session = next((s for s in self.sessions if s.id == session), None)
            if session is None:
                raise ValueError(f"Session {session} not found")

        # Initialize Inference and load model if not already done
        if not hasattr(self, 'inference') or self.inference.model is None:
            self.inference = Inference(project=self)
            self.inference.load_model(model_identifier)

        # Prepare session data
        scaled_windows = self._prepare_session_for_prediction(session)

        # Run inference
        probs = self.inference.predict(scaled_windows, batch_size=batch_size)

        # Validate predictions length
        validation_cfg = self.config.get("validation", {})
        enable_checks = validation_cfg.get("enable_checks", True)
        strict_mode = validation_cfg.get("strict_mode", False)
        
        if enable_checks and session.features is not None:
            source_length = session.features.shape[0]
            validate_predictions(probs, source_length, strict_mode=strict_mode)
        
        # Save predictions if requested
        if save:
            # Get class names for multiclass
            class_names = None
            if self.inference.task_type == "multiclass" and self.inference.class_names:
                class_names = ["background"] + self.inference.class_names

            events_df = self._probabilities_to_events(probs, threshold, class_names=class_names)
            pred_dir = session.session_root / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{session.id}_pred.csv"
            events_df.to_csv(pred_path, index=False)
            print(f"Predictions saved: {pred_path}")

            # Only create interactive plot for binary (multiclass would need different handling)
            if self.inference.task_type == "binary":
                html_path = pred_dir / f"{session.id}_interactive.html"
                self.viz.plot_interactive_predictions(probs, save_path=html_path, events_df=events_df, threshold=threshold)
        return probs

    def predict(self, sessions_to_predict=None, model_identifier=None, batch_size=64, threshold=0.5, save=True):
        """Predict on multiple sessions

        Args:
            sessions_to_predict: List of session IDs or Session objects (None uses self.sessions_to_predict)
            model_identifier: Model name or path (None for latest)
            batch_size: Batch size for inference
            threshold: Probability threshold for binary classification
            save: Whether to save predictions to CSV

        Returns:
            dict: Mapping of session_id -> probabilities array
        """
        if sessions_to_predict is None:
            sessions_to_predict = self.sessions_to_predict

        if not sessions_to_predict:
            raise ValueError("No sessions specified for prediction. Set sessions_to_predict or self.sessions_to_predict")

        # Initialize Inference once, load model, warm up
        self.inference = Inference(project=self)
        metadata = self.inference.load_model(model_identifier)

        # Warm up model
        window_size = metadata.get("window_size", self.config["data"]["window_size"])
        num_features = metadata["training_args"]["input_size"]
        self.inference.warm_up(window_size, num_features, batch_size)

        # Convert session IDs to Session objects if needed
        session_objects = []
        for sess in sessions_to_predict:
            if isinstance(sess, str):
                session_obj = next((s for s in self.sessions if s.id == sess), None)
                if session_obj is None:
                    raise ValueError(f"Session {sess} not found")
                session_objects.append(session_obj)
            else:
                session_objects.append(sess)

        # Get class names for multiclass
        class_names = None
        if self.inference.task_type == "multiclass" and self.inference.class_names:
            class_names = ["background"] + self.inference.class_names

        # Predict on each session (model already loaded, so pass None to avoid reload)
        results = {}
        validation_cfg = self.config.get("validation", {})
        enable_checks = validation_cfg.get("enable_checks", True)
        strict_mode = validation_cfg.get("strict_mode", False)
        
        for session in session_objects:
            print(f"Predicting on session: {session.id}")
            # Use the already-loaded inference instance
            scaled_windows = self._prepare_session_for_prediction(session)
            probs = self.inference.predict(scaled_windows, batch_size=batch_size)
            results[session.id] = probs

            # Validate predictions length
            if enable_checks and session.features is not None:
                source_length = session.features.shape[0]
                validate_predictions(probs, source_length, strict_mode=strict_mode)

            # Save predictions if requested
            if save:
                events_df = self._probabilities_to_events(probs, threshold, class_names=class_names)
                pred_dir = session.session_root / "predictions"
                pred_dir.mkdir(parents=True, exist_ok=True)
                pred_path = pred_dir / f"{session.id}_pred.csv"
                events_df.to_csv(pred_path, index=False)
                print(f"Predictions saved: {pred_path}")

                # Only create interactive plot for binary
                if self.inference.task_type == "binary":
                    html_path = pred_dir / f"{session.id}_interactive.html"
                    self.viz.plot_interactive_predictions(probs, save_path=html_path, events_df=events_df, threshold=threshold)

        return results

    def annotate_sessions(self):
        """Open GUI to select and annotate a session."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QApplication
        from PyQt5.QtCore import Qt
        import sys
        from behavex.annotation.pavs import start_app

        if len(self.sessions) == 0:
            print("No sessions found in project. Add sessions first using add_session().")
            return

        # Ensure QApplication exists
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Create session selection dialog
        dialog = QDialog()
        dialog.setWindowTitle("Select Session to Annotate")
        dialog.setMinimumSize(600, 400)

        layout = QVBoxLayout()
        dialog.setLayout(layout)

        label = QLabel(f"Select a session from {self.project_name}:")
        layout.addWidget(label)

        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SingleSelection)

        # Add sessions to list
        for session in self.sessions:
            item_text = f"{session.id} - {session.annotation_view} view"
            if hasattr(session, 'video_dir') and session.video_dir:
                # Show shorter path
                video_dir = str(session.video_dir)
                if len(video_dir) > 50:
                    video_dir = "..." + video_dir[-47:]
                item_text += f"\n  {video_dir}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, session)
            list_widget.addItem(item)

        # Select first item by default
        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)

        layout.addWidget(list_widget)

        # Buttons
        button_layout = QHBoxLayout()

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)

        annotate_button = QPushButton("Annotate Selected Session")
        annotate_button.setDefault(True)
        annotate_button.clicked.connect(dialog.accept)
        button_layout.addWidget(annotate_button)

        layout.addLayout(button_layout)

        # Show dialog and get selection
        if dialog.exec_() == QDialog.Accepted:
            selected_items = list_widget.selectedItems()
            if selected_items:
                selected_session = selected_items[0].data(Qt.UserRole)
                # Get annotation view path and start annotator
                try:
                    annotation_view_path = selected_session.path_to_view(selected_session.annotation_view)
                    print(f"Starting annotation for session {selected_session.id} ({selected_session.annotation_view} view)...")
                    # Pass project and sessions for session switching capability
                    # Don't run event loop since we're already in one (from dialog.exec_())
                    start_app(annotation_view_path, project=self, sessions=self.sessions, run_event_loop=False)
                except Exception as e:
                    print(f"Error starting annotation: {e}")
            else:
                print("No session selected.")
    
    def validate_session(self, session_id: str) -> dict:
        """Validate consistency for a single session.
        
        Args:
            session_id: ID of session to validate
            
        Returns:
            Dictionary with validation results
        """
        session = None
        for s in self.sessions:
            if s.id == session_id:
                session = s
                break
        
        if session is None:
            raise ValueError(f"Session {session_id} not found in project")
        
        validation_cfg = self.config.get("validation", {})
        strict_mode = validation_cfg.get("strict_mode", False)
        enable_checks = validation_cfg.get("enable_checks", True)
        
        return validate_session_consistency(
            session,
            video_frame_count=None,
            strict_mode=strict_mode,
            enable_checks=enable_checks
        )
    
    def validate_all_sessions(self) -> dict:
        """Validate consistency for all sessions in project.
        
        Returns:
            Dictionary mapping session_id -> validation results
        """
        results = {}
        validation_cfg = self.config.get("validation", {})
        strict_mode = validation_cfg.get("strict_mode", False)
        enable_checks = validation_cfg.get("enable_checks", True)
        
        for session in self.sessions:
            try:
                result = validate_session_consistency(
                    session,
                    video_frame_count=None,
                    strict_mode=strict_mode,
                    enable_checks=enable_checks
                )
                results[session.id] = result
            except Exception as e:
                results[session.id] = {
                    'is_valid': False,
                    'messages': [f"Error during validation: {e}"],
                    'warnings': [],
                    'errors': [f"Error during validation: {e}"],
                    'checks': {}
                }
        
        # Print summary
        total_sessions = len(results)
        valid_sessions = sum(1 for r in results.values() if r.get('is_valid', False))
        print(f"\nValidation Summary: {valid_sessions}/{total_sessions} sessions passed")
        
        for session_id, result in results.items():
            if not result.get('is_valid', False):
                print(f"  {session_id}: FAILED")
                for msg in result.get('warnings', []):
                    print(f"    Warning: {msg}")
                for msg in result.get('errors', []):
                    print(f"    Error: {msg}")
            elif result.get('warnings'):
                print(f"  {session_id}: PASSED (with warnings)")
                for msg in result.get('warnings', []):
                    print(f"    Warning: {msg}")
        
        return results


if __name__ == "__main__":
    # Load existing project - project_name is optional, will be loaded from config if not provided
    project_dir_root = Path("/Users/thomasbush/Downloads") / "project_dir_rear_test"
    project = Project(project_dir=str(project_dir_root))  # Name loaded from config.yaml
    # Alternative: project = Project(project_dir=str(project_dir_root), project_name="Test Project")
    
    print("\n=== Loaded project ===")
    print(f"Project directory: {project.project_dir}")
    print(f"Number of sessions: {len(project.sessions)}")
    
    # List all sessions
    print("\n=== All sessions in project ===")
    for sess in project.sessions:
        print(f"  - {sess.id}: {sess.annotation_view} view")
        print(f"    Video dir: {sess.video_dir}")
        print(f"    Has features: {sess.has_features()}")
        print(f"    Has annotations: {sess.has_annotation()}")
    
    # Import features for session 1 only
    # TODO: Update this path to your actual features file path
    session1_features_path = Path("/Users/thomasbush/Downloads/shared WithTWB/m002_s001_cricket.xlsx")
    
    if len(project.sessions) > 0:
        session1 = project.sessions[0]
        print(f"\n=== Importing features for {session1.id} ===")
        
        if session1_features_path.exists():
            try:
                imported_path = project.import_features_for_session(
                    session_id=session1.id,
                    features_path=session1_features_path
                )
                print(f"Successfully imported features to: {imported_path}")
            except Exception as e:
                print(f"Error importing features: {e}")
        else:
            print(f"Features file not found at: {session1_features_path}")
            print("Please update session1_features_path in the code")
    
    # Set feature set in config (update with your actual feature names)
    feature_set = ["height", "forepaw_tail_distance", "height_displacement", "trunk_speed"]
    project.set_config_value("data.feature_set", feature_set)
    print(f"\n=== Feature set configured ===")
    print(f"Features: {feature_set}")
    
    # Check if we can create training data
    print("\n=== Testing training data creation ===")
    try:
        # This will:
        # 1. Select features for all sessions (only session 1 has features)
        # 2. Load annotations for all sessions
        # 3. Build windows and labels
        # 4. Split data
        # 5. Merge training data
        project.process_sessions_for_training()
        
        print(f"✓ Training data created successfully!")
        print(f"  Training windows shape: {project.train_windows.shape}")
        print(f"  Training labels shape: {project.train_labels.shape}")
        print(f"  Number of training samples: {len(project.train_labels)}")
        print(f"  Positive samples: {np.sum(project.train_labels == 1)}")
        print(f"  Negative samples: {np.sum(project.train_labels == 0)}")
        
    except Exception as e:
        print(f"✗ Error creating training data: {e}")
        import traceback
        traceback.print_exc()
    # run window tuning 

    project.set_config_value("model_defaults.training.epochs", 20)
    project.tune_window_size([30, 40, 50, 60, 70, 80, 90, 100])



    # ---------------------------------------------------------
    # 4. List all sessions in this project
    # ---------------------------------------------------------

    # print("\nAll sessions in project:")
    # for sess in project.sessions:
    #     print(" -", sess.id, "at", sess.video_dir)
    #
    #
    # from behavex.annotation.pavs import start_app
    # # get session annotaiton view file path
    # annotation_view_path = session.path_to_view(session.annotation_view)
    # start_app(annotation_view_path)
    # project.import_features_for_session(
    #     session_id=session.id,
    #     features_path = Path("/Users/thomasbush/Downloads/shared WithTWB/m002_s001_cricket.xlsx"),

    # )
    # print(project.sessions[0].features_file())
    # print(project.sessions[0].events_file())
    # project.set_config_value("data.feature_set", ["height", "forepaw_tail_distance", "height_displacement", "trunk_speed"])
    # print(project.config["data"]["feature_set"])
    # # Use the new data preparation pipeline function
    # project.process_sessions_for_training()
    # print(project.train_windows.shape)
    # print(project.train_labels.shape)
    # project.set_config_value("model_defaults.training.epochs", 20)
    #
    # project.train()
    # project.train(hpo=False)
    # # Test inference
    # project.sessions_to_predict = [project.sessions[0].id]
    # results = project.predict()
    # print(f"Predictions shape for {project.sessions[0].id}: {results[project.sessions[0].id].shape}")
