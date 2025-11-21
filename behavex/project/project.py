from pathlib import Path
from typing import List, Optional

import yaml
import pandas as pd
from behavex.project.session import Session


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
        project_name: str,
    ):
        # init the project directory
        self.project_dir = Path(project_dir)
        self.project_name = project_name
        self.sessions_path = self.project_dir / "sessions.yaml"
        if not self.project_dir.exists():
            self.project_dir.mkdir(parents=True)

        # load or create config file
        self.load_config()
        self.load_sessions()

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
            "model_defaults": {
                "model_name": "gru_v1",
                "training": {
                    "epochs": 20,
                    "batch_size": 32,
                    "lr": 1e-3,
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
                    start_app(annotation_view_path, project=self, sessions=self.sessions)
                except Exception as e:
                    print(f"Error starting annotation: {e}")
            else:
                print("No session selected.")


if __name__ == "__main__":
    # simple test of the Project class
    project_dir_root = Path("/users/thomasbush/Downloads") / "project_dir_rear"
    project = Project(project_dir=str(project_dir_root), project_name="Test Project")
    project.set_config_value("data.window_size", 40)
    updates = {
        "data": {
            "window_size": 40,
            "fps": 60,
        },
        "model_defaults": {
            "training": {
                "epochs": 30,
                "batch_size": 64,
            }
        },
    }
    project.edit_config(updates)
    project.show_config()

    session_path = Path(
        "/Users/thomasbush/Downloads/multicam_video_2025-05-07T12_16_20_cropped-v2_20250701121021"
    )

    # This automatically:
    #   - detects the 5 camera files
    #   - maps them to views (central, bottom, left, right, top)
    #   - adds metadata to sessions.yaml
    #   - creates Session object
    #   - creates folder: project_dir/sessions/session_XXX/
    session = project.add_session(session_path)

    print(f"\nAdded session: {session.id}")
    print("Views detected:", list(session.views.keys()))
    print("Annotation view:", session.annotation_view)
    print("Annotation file expected at:", session.annotation_path())
    print("Feature file path:", session.features_path())

    # ---------------------------------------------------------
    # 4. List all sessions in this project
    # ---------------------------------------------------------

    print("\nAll sessions in project:")
    for sess in project.sessions:
        print(" -", sess.id, "at", sess.video_dir)


    # from behavex.annotation.pavs import start_app
    # # get session annotaiton view file path 
    # annotation_view_path = session.path_to_view(session.annotation_view)
    # start_app(annotation_view_path)
    # project.import_features_for_session(
    #     session_id=session.id,
    #     features_path = Path("/Users/thomasbush/Downloads/shared WithTWB/m002_s001_cricket.xlsx"),

    # )
    # print(project.sessions[0].features_file())
    print(project.sessions[0].events_file())
    # project.annotate_sessions()
    project.set_config_value("data.feature_set", ["height", "dist"])
    print(project.config["data"]["feature_set"])
    project.select_features()
    print(project.sessions[0].features.shape)
    project.sessions[0].load_annotations()
    print(project.sessions[0].annotations.head())
    project.sessions[0].build_labels()
    print(project.sessions[0].labels.shape)
    project.sessions[0].build_windows()
    print(project.sessions[0].windows.shape)
    project.sessions[0].subsample_session()
    print(project.sessions[0].windows.shape)
    print(project.sessions[0].labels.shape)

    project.sessions[0].split_data()
    print(project.sessions[0].train_windows.shape)
    print(project.sessions[0].train_labels.shape)
    print(project.sessions[0].val_windows.shape)
    print(project.sessions[0].val_labels.shape)