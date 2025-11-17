import csv
from pathlib import Path

import napari
from napari.utils.notifications import show_info

CSV_OUT = Path('~/Desktop/data.csv').expanduser()
if not CSV_OUT.exists():
    CSV_OUT.write_text("file,frame,action\n")

KEYMAP = {
    'h': 'start_rearing',
    'j': 'stop_rearing',
}

viewer = napari.Viewer()

def on_keypress(key, viewer):
    action = KEYMAP[key]
    frame = viewer.dims.current_step[0]
    layer = viewer.layers.selection.active or viewer.layers[-1]

    show_info(action)
    with open(CSV_OUT, 'a') as f:
        csv.writer(f).writerow([layer.source.path, frame, action])

def make_callback(key):
    # this function *does* have a __name__
    def _cb(viewer):
        on_keypress(key, viewer)
    return _cb

for key in KEYMAP:
    viewer.bind_key(key, make_callback(key))

napari.run()

