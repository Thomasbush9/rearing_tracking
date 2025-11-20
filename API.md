# BehaveX API Documentation

API reference for `behavex` modules: Project, Session, Annotation, and Features.

## Project Module

### `Project`

Manages behavioral analysis projects.

#### Initialization

```python
from behavex.project import Project

project = Project(
    project_dir: str,
    project_name: str
)
```

Creates or loads a project directory with config and session management.

#### Methods

##### `load_config()`
Loads or creates `config.yaml` with default structure.

##### `load_sessions()`
Loads `sessions.yaml` and creates `Session` objects. Auto-merges duplicates.

##### `add_session(session_dir: Path, session_id: str = None, annotation_view: str = None) -> Session`
Adds a session to the project.

**Parameters:**
- `session_dir`: Directory containing video files (.mp4)
- `session_id`: Optional ID (auto-generated if None)
- `annotation_view`: View name for annotation (uses first view if None)

**Returns:** `Session` instance

**Behavior:**
- Detects video files and extracts view names from filenames
- Merges if session with same path already exists
- Creates session directory at `project_dir/sessions/{session_id}/`

##### `save_sessions()`
Writes current session registry to `sessions.yaml`.

##### `set_config_value(key_path: str, value, save: bool = True)`
Sets a single config value using dot notation.

**Example:**
```python
project.set_config_value("data.window_size", 40)
```

##### `edit_config(updates: dict, save: bool = True)`
Recursively updates config with provided dict. Only specified keys are changed.

**Example:**
```python
project.edit_config({
    "data": {"window_size": 40, "fps": 60},
    "model_defaults": {"training": {"epochs": 30}}
})
```

##### `show_config()`
Prints current config to console.

##### `import_features_for_session(session_id: str, features_path: str | Path, video_frame_count: int = None) -> Path`
Imports features file for a session.

**Parameters:**
- `session_id`: Session ID (e.g., "session_001")
- `features_path`: Path to CSV or Excel file
- `video_frame_count`: Optional frame count for validation

**Returns:** Path to imported features file

##### `annotate_sessions()`
Opens GUI dialog to select and annotate a session. Uses PyQt5.

#### Attributes

- `project_dir: Path` - Project root directory
- `project_name: str` - Project name
- `config: dict` - Configuration dictionary
- `sessions: List[Session]` - List of session objects
- `sessions_data: List[dict]` - Raw session metadata

---

## Session Module

### `Session`

Represents a single behavioral recording session.

#### Initialization

```python
from behavex.project.session import Session

# Usually created via Project.add_session()
session = Session(metadata: dict, project: Project)
```

#### Methods

##### `path_to_view(view_name: str) -> Path`
Returns absolute path to video for specified view.

**Raises:** `KeyError` if view not found

##### `annotation_path() -> Path`
Returns expected path for annotation CSV: `project_dir/sessions/{session_id}/{session_id}_annotations.csv`

##### `features_path() -> Path`
Returns path for features CSV. Checks metadata for custom path, otherwise defaults to `{session_id}_features.csv` in session directory.

##### `windows_path() -> Path`
Returns path for windowed features: `{session_id}_windows.npy`

##### `predictions_path() -> Path`
Returns path for model predictions: `{session_id}_predictions.npy`

##### `has_annotation() -> bool`
Checks if annotation file exists.

##### `has_features() -> bool`
Checks if features file exists.

##### `import_features(features_path: Path, video_frame_count: int = None) -> Path`
Imports features from CSV or Excel file.

**Parameters:**
- `features_path`: Path to .csv, .xlsx, or .xls file
- `video_frame_count`: Optional frame count for validation

**Returns:** Path to imported CSV file

**Behavior:**
- Converts Excel to CSV
- Saves to session directory
- Updates session metadata

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If format is invalid

##### `has_windows() -> bool`
Checks if windowed data exists.

##### `has_predictions() -> bool`
Checks if inference output exists.

##### `update_metadata(key, value)`
Updates a metadata field. Project will save to YAML on next `save_sessions()`.

#### Attributes

- `id: str` - Session identifier
- `video_dir: Path` - Directory containing video files
- `views: dict` - Dictionary mapping view names to video paths
- `annotation_view: str` - View name used for annotation
- `session_root: Path` - Session directory in project
- `project: Project` - Parent project instance
- `metadata: dict` - Raw session metadata dictionary

---

## Annotation Module

### `start_app(video_path: Path = None, project: Project = None, sessions: List[Session] = None)`

Launches PyQt5 annotation GUI for behavioral events.

**Parameters:**
- `video_path`: Optional video file to auto-load
- `project`: Optional Project instance for session switching
- `sessions`: Optional list of sessions for session switching

**Usage:**
```python
from behavex.annotation.pavs import start_app
from pathlib import Path

# Simple usage
start_app(Path("/path/to/video.mp4"))

# With project context (enables session switching)
start_app(Path("/path/to/video.mp4"), project=project, sessions=project.sessions)
```

### Annotation Window Features

#### Video Controls
- Play/pause
- Frame navigation (arrow keys)
- Frame slider
- Time/frame display

#### Event Annotation
- **Keyboard shortcuts:**
  - `h` - Mark rearing start
  - `j` - Mark rearing end
  - `Space` - Play/pause
  - `←/→` - Move 1 frame
  - `Shift + ←/→` - Move 10 frames
  - `L` - Open file
  - `C` - Clear all events

#### CSV Export
- Auto-saves on event completion
- Format: `index, start_frame, end_frame, duration`
- Saved as `{video_stem}_rearings.csv` in video directory

#### Event Visualization
- Timeline plot with event intervals (red bars)
- Current frame indicator (blue line)
- Zoom controls (zoom in/out, reset)
- Click plot to jump to frame

#### Feature Integration
- Import features CSV/Excel via "Import Features" button
- Select features to plot via checkboxes
- Separate feature plot window showing selected features with event overlays
- Auto-loads features if available for session

#### Session Management (if project provided)
- "Switch Session" button to change between sessions
- Auto-loads annotations and features when switching

---

## Features Module

### Current Status

The `behavex.features` module is currently minimal. Feature data is managed through:

1. **Session-level import:** `Session.import_features()` or `Project.import_features_for_session()`
2. **Annotation integration:** Features can be loaded and visualized in the annotation GUI
3. **Storage:** Features stored as CSV in session directory at `{session_id}_features.csv`

### Feature File Format

- **Supported formats:** CSV, Excel (.xlsx, .xls)
- **Structure:** Each column is a feature, each row is a frame
- **Validation:** Length validation optional (not enforced by default)
- **NaN handling:** NaN values allowed (features may have missing values)

### Usage Example

```python
# Import features via project
project.import_features_for_session(
    "session_001",
    "/path/to/features.xlsx"
)

# Or directly via session
session = project.sessions[0]
session.import_features("/path/to/features.csv")

# Features automatically available in annotation GUI
project.annotate_sessions()
```

---

## Configuration Structure

Default `config.yaml` structure:

```yaml
project:
  name: str
  version: str
  description: str

paths:
  video_root: str
  keypoints_root: str

data:
  fps: int
  window_size: int
  feature_set: list

annotation:
  behavior: str
  annotator: str
  output_format: str

model_defaults:
  model_name: str
  training:
    epochs: int
    batch_size: int
    lr: float
  inference:
    smoothing: bool
    smoothing_window: int

debug:
  verbose: bool
  save_intermediate: bool
```

---

## Session Metadata Structure

`Sessions` are stored in `sessions.yaml`:

```yaml
sessions:
  - id: str  # e.g., "session_001"
    video_dir: str  # Absolute path to video directory
    views:
      view_name: str  # e.g., "bottom": "/path/to/video.mp4"
    annotation_view: str  # View name used for annotation
    features_file: str  # Optional: relative path to features CSV
    date: str  # Optional
    animal_id: str  # Optional
    group: str  # Optional
```

