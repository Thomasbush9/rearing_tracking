from pathlib import Path


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
        """Return path for the extracted feature matrix."""
        return self.session_root / f"{self.id}_features.npy"

    def windows_path(self) -> Path:
        """Return path for windowed features."""
        return self.session_root / f"{self.id}_windows.npy"

    def predictions_path(self) -> Path:
        """Return path for model predictions."""
        return self.session_root / f"{self.id}_predictions.npy"

    # ------------------------------------------------------------
    # State checks
    # ------------------------------------------------------------

    def has_annotation(self) -> bool:
        """Check whether annotations exist."""
        return self.annotation_path().exists()

    def has_features(self) -> bool:
        """Check if extracted features exist."""
        return self.features_path().exists()

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
