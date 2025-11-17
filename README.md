# Rearing Annotator

## Installation

```bash
pip install napari qtpy
```

## Usage

1. Run the script: `python rearing_annotator.py`
2. Load your video in napari (File → Open Files)
3. Press `h` to mark the start of a rearing event
4. Press `j` to mark the end of a rearing event
5. Events are displayed in the right panel widget
6. CSV file (`{video_name}_rearings.csv`) is saved in the same directory as your video when each event completes

**Keyboard shortcuts:**
- `h` - Start rearing event
- `j` - End rearing event

The widget shows: Index, Start frame, End frame, Duration (in frames).
