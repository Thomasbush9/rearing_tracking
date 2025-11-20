from pathlib import Path
import csv
import shutil
import pandas as pd
import numpy as np


class Session:
    """
    Represents a single behavioral recording session inside a Project.
    Holds metadata and provides convenient path utilities.
    """

    def __init__(self, metadata: dict, project):
        """
        Args:
            metadata: dict describing the session (from sessions.yaml)
            project: Project instance owning this session
        """
        self.project = project
        self.metadata = metadata  # raw dict; Project writes it back

        # Required fields
        self.id = metadata.get("id")
        self.video_dir = Path(metadata.get("video_dir", ""))
        self.views = metadata.get("views", {})
        self.annotation_view = metadata.get("annotation_view", None)

        # Optional fields
        self.date = metadata.get("date", None)
        self.animal_id = metadata.get("animal_id", None)
        self.group = metadata.get("group", None)

        # Directory inside project for artifacts
        self.session_root = self.project.project_dir / "sessions" / self.id

        self.features = None
        self.annotations = None
        self.labels = None
    # ------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------

    def path_to_view(self, view_name: str) -> Path:
        """Return absolute path to a video for the given view."""
        if view_name not in self.views:
            raise KeyError(f"View '{view_name}' not found in session {self.id}.")
        return Path(self.views[view_name]).expanduser().resolve()

    def annotation_path(self) -> Path:
        """Return expected path for the annotation CSV."""
        if self.annotation_view is None:
            raise ValueError(f"Session {self.id} does not define annotation_view.")

        # This stays inside project/session-root
        return self.session_root / f"{self.id}_annotations.csv"

    def features_path(self) -> Path:
        """Return path for the extracted feature CSV."""
        # Check if CSV exists in metadata, otherwise default to CSV
        features_file = self.metadata.get('features_file', None)
        if features_file:
            # Check if it's an absolute path or relative
            features_path = Path(features_file)
            if features_path.is_absolute():
                return features_path
            else:
                return self.session_root / features_file
        return self.session_root / f"{self.id}_features.csv"

    def windows_path(self) -> Path:
        """Return path for windowed features."""
        return self.session_root / f"{self.id}_windows.npy"

    def predictions_path(self) -> Path:
        """Return path for model predictions."""
        return self.session_root / f"{self.id}_predictions.npy"

    def events_file(self) -> Path:
        """Return path to events CSV in video_dir. Looks for CSV matching video filename, then annotation_view.csv."""
        if not self.video_dir:
            raise ValueError(f"Session {self.id} video_dir is not set")
        video_dir = Path(self.video_dir).expanduser().resolve()
        if not video_dir.exists():
            raise ValueError(f"Session {self.id} video_dir does not exist: {video_dir}")
        
        # First, try to find CSV matching the annotation view video filename
        if self.annotation_view:
            try:
                video_path = self.path_to_view(self.annotation_view)
                video_stem = video_path.stem
                # Look for CSV with same name as video (e.g., video.mp4 -> video.csv)
                video_based_csv = video_dir / f"{video_stem}.csv"
                if video_based_csv.exists():
                    return video_based_csv
                # Also check for _rearings suffix pattern
                rearings_csv = video_dir / f"{video_stem}_rearings.csv"
                if rearings_csv.exists():
                    return rearings_csv
            except (KeyError, ValueError):
                pass  # Fall through to default
        
        # Fall back to annotation_view.csv
        events_path = video_dir / "annotation_view.csv"
        # Initialize empty CSV if it doesn't exist
        if not events_path.exists():
            self._initialize_events_file(events_path)
        return events_path

    def features_file(self) -> Path:
        """Return path to features file."""
        return self.features_path()

    def _initialize_events_file(self, events_path: Path):
        """Initialize an empty events CSV file with header if it doesn't exist."""
        if events_path.exists():
            return
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(events_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'start_frame', 'end_frame', 'duration'])

    # ------------------------------------------------------------
    # State checks
    # ------------------------------------------------------------

    def has_annotation(self) -> bool:
        """Check whether annotations exist."""
        return self.annotation_path().exists()

    def has_features(self) -> bool:
        """Check if extracted features exist."""
        return self.features_path().exists()
    
    def import_features(self, features_path: Path, video_frame_count: int = None) -> Path:
        """
        Import features from a CSV or Excel file.
        
        Args:
            features_path: Path to the CSV or Excel file (.csv, .xlsx, .xls) containing features
            video_frame_count: Expected number of frames (for validation). 
                              If None, will try to estimate from video or skip validation.
        
        Returns:
            Path to the imported features file (always saved as CSV)
        
        Raises:
            FileNotFoundError: If features_path doesn't exist
            ValueError: If file format is invalid or length doesn't match video
        """
        features_path = Path(features_path).expanduser().resolve()
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        # Determine file type and read accordingly
        file_ext = features_path.suffix.lower()
        
        try:
            if file_ext in ['.xlsx', '.xls']:
                # Read Excel file
                df = pd.read_excel(features_path)
                feature_length = len(df)
                num_features = len(df.columns)
                # Convert to CSV for storage
                feature_data = df
            elif file_ext == '.csv':
                # Read CSV file
                df = pd.read_csv(features_path)
                feature_length = len(df)
                num_features = len(df.columns)
                feature_data = df
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. Supported: .csv, .xlsx, .xls")
            
            # Note: We allow NaN values in features (different columns can have different numbers of valid values)
            # This is common in behavioral data where some features may be unavailable for certain frames
        
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty or has no data")
        except Exception as e:
            if isinstance(e, ValueError) and ("Unsupported" in str(e) or "length" in str(e)):
                raise
            raise ValueError(f"Error reading features file: {e}")
        
        # Skip validation - allow features of any length
        # Note: video_frame_count parameter kept for API compatibility but validation is disabled
        
        # Ensure session directory exists
        self.session_root.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV in session directory (always convert to CSV for consistency)
        target_path = self.session_root / f"{self.id}_features.csv"
        feature_data.to_csv(target_path, index=False)
        
        # Update metadata
        self.metadata['features_file'] = f"{self.id}_features.csv"
        self.project.save_sessions()
        
        print(f"Imported {num_features} features ({feature_length} frames) from {features_path.name} to {target_path}")
        return target_path

    def has_windows(self) -> bool:
        """Check if windowed data exists."""
        return self.windows_path().exists()

    def has_predictions(self) -> bool:
        """Check if inference output exists."""
        return self.predictions_path().exists()

    # ------------------------------------------------------------
    # Updating metadata (Project will write YAML)
    # ------------------------------------------------------------

    def update_metadata(self, key, value):
        """Update a field in metadata (Project should save YAML)."""
        self.metadata[key] = value


    def select_features(self, features_set: list):
        """Load features from file and select the features in the features_set."""
        features_file = self.features_file()
        if self.has_features():
            if features_file.suffix == ".csv":
                features = pd.read_csv(features_file)
            elif features_file.suffix == ".xlsx":
                features = pd.read_excel(features_file)
            else:
                raise ValueError(f"Unsupported file format: {features_file.suffix}")
            try: 
                self.features = features[features_set].to_numpy()
            except Exception as e:
                raise ValueError(f"Error selecting features: {e}")

    def load_annotations(self):
        """
        Load annotation/events file as pandas DataFrame.
        Loads from video_dir/annotation_view.csv (same file as events_file()).
        File is initialized if it doesn't exist.
        """
        events_path = self.events_file()  # This initializes the file if missing
        self.annotations = pd.read_csv(events_path)
    
    def build_labels(self):
        """Build labels from annotaions"""
        if self.annotations is None:
            raise ValueError("Annotations not loaded. Call load_annotations() first.")
        if self.features is None:
            raise ValueError("Features not loaded. Call select_features() first.")
        y = np.zeros(self.features.shape[0])
        if "start_frame" not in self.annotations.columns or "end_frame" not in self.annotations.columns:
            raise ValueError("Annotations must contain start_frame and end_frame columns.")
        for s, e in self.annotations[["start_frame", "end_frame"]].itertuples(index=False):
            y[s:e] = 1
        self.labels = y