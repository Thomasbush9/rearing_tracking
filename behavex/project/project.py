from pathlib import Path
from typing import List, Optional

import yaml


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

            return
        default_data = {"sessions": []}
        with open(self.sessions_path, "w") as stream:
            yaml.safe_dump(default_data, stream, sort_keys=False)
        self.sessions_data = []
        self.sessions = []

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

    def add_session(
        self,
    ):
        pass

    def get_session(
        self,
    ):
        pass


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
