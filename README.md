# BehaveX - Behavior Analysis Toolkit

A Python toolkit for annotating, training, and predicting animal behaviors from video and pose estimation data.

## Installation

### Using UV (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies and install the package
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install the package in editable mode
uv pip install -e .
```

### Using pip (Alternative)

```bash
pip install -e .
```

## Quick Start

```python
from behavex.project.project import Project
from pathlib import Path

# Create or load a project
project = Project("/path/to/my_project")
```

---

## Project API Guide

### 1. Create a Project

```python
from behavex.project.project import Project
from pathlib import Path

# Create a new project (or load existing)
project = Project(
    project_dir="/path/to/my_behavior_project",
    project_name="Mouse Rearing Study"  # Optional, uses dir name if not provided
)
```

This creates:
- `config.yaml` - Project configuration
- `sessions.yaml` - Session registry

---

### 2. Add Sessions

Each session links a video directory to features and labels.

```python
# Add a session from a directory containing .mp4 video files
session = project.add_session(
    session_dir="/path/to/video_folder",
    session_id="mouse_001",  # Optional, auto-generated if not provided
    annotation_view="bottom"  # Which camera view to annotate (default: "bottom")
)

# Add multiple sessions
for video_dir in Path("/data/videos").iterdir():
    if video_dir.is_dir():
        project.add_session(video_dir)

# List all sessions
for session in project.sessions:
    print(f"{session.id}: {session.video_dir}")
```

---

### 3. Configure Behaviors (Labels)

Set custom behavior labels for your project:

```python
# Set behaviors for multiclass annotation
project.set_config_value("annotation.behaviors", [
    "rearing",
    "grooming",
    "exploring",
    "resting"
])

# For binary classification (single behavior)
project.set_config_value("annotation.behavior", "rearing")
```

Default behaviors: `["rearing", "grooming", "exploring", "resting", "other"]`

---

### 4. Import Features

Import pose estimation features (CSV or Excel) for each session:

```python
# Import features for a specific session
project.import_features_for_session(
    session_id="mouse_001",
    features_path="/path/to/features.xlsx"
)

# Or import directly on session object
session = project.sessions[0]
session.import_features(Path("/path/to/features.csv"))
```

---

### 5. Annotate Sessions (Label Data)

#### Option A: Use the GUI Annotator

```python
# Open annotation GUI with session selector
project.annotate_sessions()
```

#### Option B: Direct Video Annotation

```python
from behavex.annotation.pavs import start_app

# Annotate a specific video
session = project.sessions[0]
video_path = session.path_to_view(session.annotation_view)
start_app(video_path, project=project, sessions=project.sessions)
```

**Annotation Hotkeys:**
- `H` - Start behavior event
- `J` - End behavior event
- `1-9` - Quick select behavior type
- `Space` - Play/Pause
- `Left/Right` - Frame step
- `Shift+Left/Right` - 10 frame step

Labels are auto-saved to `{video_name}_rearings.csv` in the video directory.

---

### 6. Configure Feature Set for Training

```python
# Set which features to use for model training
project.set_config_value("data.feature_set", [
    "height",
    "forepaw_tail_distance",
    "height_displacement",
    "trunk_speed"
])

# Set window size (frames of context)
project.set_config_value("data.window_size", 30)
```

---

### 7. Train the Model

```python
# Process all sessions and prepare training data
project.process_sessions_for_training()

# Configure training parameters
project.set_config_value("model_defaults.training.epochs", 50)
project.set_config_value("model_defaults.training.device", "cuda")  # or "mps", "cpu"
project.set_config_value("model_defaults.training.batch_size", 32)
project.set_config_value("model_defaults.training.hidden_size", 32)
project.set_config_value("model_defaults.training.lr", 0.001)

# Train the model
project.train()

# Or with hyperparameter optimization
project.train(hpo=True, n_trials=20)
```

**Multiclass Training:**

```python
# Configure for multiclass
project.set_config_value("model_defaults.training.task_type", "multiclass")
project.set_config_value("annotation.behaviors", ["rearing", "grooming", "exploring"])

project.process_sessions_for_training()
project.train()
```

### Transformer Training Script (Masked Modeling)

For large-scale masked modeling experiments, use the standalone trainer at [`src/behavex/models/train_transformer.py`](src/behavex/models/train_transformer.py):

```bash
python src/behavex/models/train_transformer.py \
  --data /path/to/m001_s001_cricket.xlsx \
  --checkpoint-every 5 \
  --resume-from /runs/masked_transformer_20260202/best.pt \
  --use-compile \
  --wandb-project rearing-transformer \
  --wandb-tags long_run apples
```

Highlights:
- `best.pt`, `last.pt`, and optional `checkpoints/checkpoint_epochN.pt` store model, optimizer, epoch, patience, and RNG state so `--resume-from` can recover full training context.
- `--use-compile` (plus `--compile-backend` / `--compile-mode`) leverages `torch.compile` for faster throughput on PyTorch 2.x.
- W&B logging is opt-in via `--wandb-project`; metrics, reconstruction plots, and config sync automatically (install with `pip install wandb`). Use `--wandb-mode offline` for air-gapped runs.
- TensorBoard logging remains enabled inside each run directory.

Run `python src/behavex/models/train_transformer.py --help` for the full list of arguments.

---

### 8. Run Predictions

```python
# Set sessions to predict on
project.sessions_to_predict = ["mouse_001", "mouse_002"]

# Run predictions (uses latest trained model)
results = project.predict(threshold=0.5)

# Or predict on a single session
probs = project.predict_session("mouse_001")

# Predictions saved to: {session_root}/predictions/{session_id}_pred.csv
```

---

### 9. Tune Window Size

```python
# Find optimal window size
results = project.tune_window_size([30, 40, 50, 60, 80, 100])

# View results in TensorBoard
# tensorboard --logdir {project_dir}/models/runs
```

---

## Complete Workflow Example

```python
from behavex.project.project import Project
from pathlib import Path

# 1. Create project
project = Project("/data/my_rearing_project")

# 2. Add sessions
for video_dir in Path("/data/experiment_videos").glob("mouse_*"):
    project.add_session(video_dir)

# 3. Configure behaviors for this project
project.set_config_value("annotation.behaviors", [
    "rearing",
    "grooming",
    "exploring"
])

# 4. Import features for each session
features_dir = Path("/data/pose_features")
for session in project.sessions:
    features_file = features_dir / f"{session.id}_features.xlsx"
    if features_file.exists():
        project.import_features_for_session(session.id, features_file)

# 5. Annotate sessions (opens GUI)
project.annotate_sessions()

# 6. Configure and train
project.set_config_value("data.feature_set", [
    "height", "forepaw_tail_distance", "height_displacement", "trunk_speed"
])
project.set_config_value("data.window_size", 50)
project.set_config_value("model_defaults.training.epochs", 30)

project.process_sessions_for_training()
project.train()

# 7. Predict on new sessions
project.sessions_to_predict = [s.id for s in project.sessions]
project.predict()
```

---

## Project Structure

```
my_project/
├── config.yaml          # Project configuration
├── sessions.yaml        # Session registry
├── sessions/            # Session data (auto-created)
│   └── {session_id}/
│       ├── {id}_features.csv
│       ├── {id}_annotations.csv
│       └── predictions/
│           ├── {id}_pred.csv
│           └── {id}_interactive.html
└── models/              # Trained models
    ├── runs/            # TensorBoard logs
    └── {model_name}/
        ├── model.pt
        └── metadata.yaml
```

---

## Validation

The project includes automatic consistency validation between videos, features, labels, and predictions:

```python
# Validate all sessions
results = project.validate_all_sessions()

# Validate a specific session
result = project.validate_session("session_001")

# Configure validation behavior
project.set_config_value("validation.enable_checks", True)  # Enable/disable checks
project.set_config_value("validation.strict_mode", False)  # Raise errors vs warnings
```

Validation checks:
- **Video ↔ Features**: Features length matches video frame count (1% tolerance)
- **Features ↔ Annotations**: Annotation frame ranges are within valid bounds
- **Features ↔ Labels**: Labels length exactly matches features length
- **Features ↔ Predictions**: Predictions length matches source data length

By default, validation runs in warning mode (prints warnings but continues execution). Set `strict_mode: true` to raise errors on mismatches.

## Configuration Reference

Key `config.yaml` settings:

```yaml
project:
  name: "My Project"
  version: "1.0"

data:
  fps: 60                    # Video frame rate
  window_size: 30            # Context window (frames)
  feature_set: []            # Features for training

annotation:
  behavior: "rearing"        # Single behavior (binary mode)
  behaviors: []              # Multiple behaviors (multiclass mode)

model_defaults:
  training:
    epochs: 20
    device: "mps"            # "cuda", "mps", or "cpu"
    batch_size: 32
    hidden_size: 16
    lr: 0.001
    task_type: "binary"      # or "multiclass"

validation:
  enable_checks: true        # Enable consistency validation
  strict_mode: false          # Raise errors vs warnings
```

---

## Keyboard Shortcuts (Annotation App)

| Key | Action |
|-----|--------|
| `H` | Start behavior event |
| `J` | End behavior event |
| `1-9` | Select behavior type |
| `Space` | Play/Pause video |
| `Left/Right` | Step 1 frame |
| `Shift+Left/Right` | Step 10 frames |
| `+/-` | Zoom in/out on plot |
| `L` | Open video file |
| `C` | Clear all events |
