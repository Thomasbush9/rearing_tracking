import csv
from pathlib import Path

import napari
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QLabel, QApplication, QPushButton
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QBrush

# adjust keybindings to your liking
KEYMAP = {
    'h': 'rearing_start',
    'j': 'rearing_end',
}

viewer = napari.Viewer()

# track events
events_coords = []  # list of frame numbers
events_actions = []  # list of 'rearing_start' or 'rearing_end'

# track if there's an open rearing event
open_rearing = False

# CSV file path (will be set when video is loaded)
CSV_OUT = None

# Navigation speed (step size in frames)
nav_step_size = 1

# Playback fps
playback_fps = 60

def get_csv_path():
    """Get CSV path based on current video file."""
    global CSV_OUT
    # Find the video/image layer
    for layer in viewer.layers:
        if hasattr(layer, 'source') and hasattr(layer.source, 'path'):
            video_path = Path(layer.source.path)
            # Create CSV path: same directory, {video_name}_rearings.csv
            CSV_OUT = video_path.parent / f"{video_path.stem}_rearings.csv"
            return CSV_OUT
    return None

def write_csv():
    """Write all complete events to CSV file."""
    csv_path = get_csv_path()
    if csv_path is None:
        return
    
    # Group events into complete pairs
    events = []
    i = 0
    while i < len(events_coords):
        if i < len(events_actions) and events_actions[i] == 'rearing_start':
            start_frame = int(events_coords[i])
            # find corresponding end
            end_frame = None
            for j in range(i + 1, len(events_coords)):
                if j < len(events_actions) and events_actions[j] == 'rearing_end':
                    end_frame = int(events_coords[j])
                    break
            if end_frame is not None:
                duration = end_frame - start_frame
                events.append((start_frame, end_frame, duration))
            i += 1
        else:
            i += 1
    
    # Write complete CSV file
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'start_frame', 'end_frame', 'duration'])
        for idx, (start, end, duration) in enumerate(events, 1):
            writer.writerow([idx, start, end, duration])

# Events widget to display on the side
class EventsWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setLayout(QVBoxLayout())
        self.events_list = []  # store events for display
        
        # Make widget visible with background
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.setMinimumSize(350, 500)
        
        title = QLabel("Rearing Events")
        title.setStyleSheet("font-weight: bold; font-size: 16px; padding: 5px; color: white;")
        self.layout().addWidget(title)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Index", "Start", "End", "Duration"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setMinimumHeight(300)
        self.table.setMinimumWidth(300)
        self.table.setRowCount(0)
        self.table.setShowGrid(True)
        self.table.setAlternatingRowColors(True)
        # Style the table
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: white;
                gridline-color: #555555;
            }
            QHeaderView::section {
                background-color: #3d3d3d;
                color: white;
                padding: 4px;
                border: 1px solid #555555;
            }
        """)
        self.layout().addWidget(self.table)
        
        # Add a status label
        self.status_label = QLabel("No events yet")
        self.status_label.setStyleSheet("color: white; padding: 5px;")
        self.layout().addWidget(self.status_label)
        
        # Navigation controls
        nav_label = QLabel("Navigation Speed")
        nav_label.setStyleSheet("color: white; padding: 5px; font-weight: bold;")
        self.layout().addWidget(nav_label)
        
        self.nav_speed_label = QLabel(f"Step: {nav_step_size} frames")
        self.nav_speed_label.setStyleSheet("color: white; padding: 2px;")
        self.layout().addWidget(self.nav_speed_label)
        
        # Buttons for speed control
        btn_layout = QVBoxLayout()
        btn_faster = QPushButton("Faster [+]")
        btn_faster.setStyleSheet("padding: 5px;")
        btn_faster.clicked.connect(self.increase_speed)
        btn_layout.addWidget(btn_faster)
        
        btn_slower = QPushButton("Slower [-]")
        btn_slower.setStyleSheet("padding: 5px;")
        btn_slower.clicked.connect(self.decrease_speed)
        btn_layout.addWidget(btn_slower)
        
        btn_widget = QWidget()
        btn_widget.setLayout(btn_layout)
        self.layout().addWidget(btn_widget)
        
        # 60fps playback button
        self.fps_btn = QPushButton(f"Set Playback: {playback_fps} fps")
        self.fps_btn.setStyleSheet("padding: 5px; margin-top: 10px;")
        self.fps_btn.clicked.connect(self.set_60fps)
        self.layout().addWidget(self.fps_btn)
    
    def increase_speed(self):
        """Increase navigation step size."""
        global nav_step_size
        nav_step_size = min(nav_step_size * 2, 1000)
        self.nav_speed_label.setText(f"Step: {nav_step_size} frames")
        show_info(f"Navigation speed: {nav_step_size} frames")
    
    def decrease_speed(self):
        """Decrease navigation step size."""
        global nav_step_size
        nav_step_size = max(nav_step_size // 2, 1)
        self.nav_speed_label.setText(f"Step: {nav_step_size} frames")
        show_info(f"Navigation speed: {nav_step_size} frames")
    
    def set_60fps(self):
        """Set playback to 60fps."""
        global playback_fps
        playback_fps = 60
        self.fps_btn.setText(f"Set Playback: {playback_fps} fps")
        # Set fps using napari's dims play rate
        # Rate is in fps - convert to frames per second
        try:
            # napari uses rate parameter - 60 fps means 60 frames per second
            # If playback is running, update it
            if viewer.dims.is_playing:
                viewer.dims.play(axis=0, fps=playback_fps)
            # Also set on video layers if available
            for layer in viewer.layers:
                # Try different attributes napari might use
                if hasattr(layer, 'fps'):
                    layer.fps = playback_fps
                elif hasattr(layer, 'play'):
                    # Some layers use play rate
                    if hasattr(layer, 'play_rate'):
                        layer.play_rate = playback_fps / 30.0  # Normalize if needed
        except:
            pass
        show_info(f"Playback set to {playback_fps} fps")
        
    def update_table(self, events_coords, events_actions):
        """Update the table with current events."""
        # Group start/end pairs
        events = []
        i = 0
        while i < len(events_coords):
            if i < len(events_actions) and events_actions[i] == 'rearing_start':
                start_frame = int(events_coords[i])
                # find corresponding end
                end_frame = None
                for j in range(i + 1, len(events_coords)):
                    if j < len(events_actions) and events_actions[j] == 'rearing_end':
                        end_frame = int(events_coords[j])
                        break
                if end_frame is not None:
                    duration = end_frame - start_frame
                    events.append((start_frame, end_frame, duration))
                else:
                    # incomplete event (start but no end yet)
                    events.append((start_frame, None, None))
                i += 1
            else:
                i += 1
        
        self.events_list = events
        
        # Update status label
        if len(events) == 0:
            self.status_label.setText("No events yet")
        else:
            complete = sum(1 for e in events if e[1] is not None)
            self.status_label.setText(f"{len(events)} event(s) - {complete} complete")
        
        # Clear and repopulate table
        self.table.setRowCount(0)
        self.table.setRowCount(len(events))
        
        for row, (start, end, duration) in enumerate(events):
            # Index column
            item_idx = QTableWidgetItem(str(row + 1))
            item_idx.setFlags(item_idx.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item_idx)
            
            # Start column
            item_start = QTableWidgetItem(str(start))
            item_start.setFlags(item_start.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, item_start)
            
            if end is not None:
                # End column
                item_end = QTableWidgetItem(str(end))
                item_end.setFlags(item_end.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 2, item_end)
                
                # Duration column
                item_dur = QTableWidgetItem(f"{duration} frames")
                item_dur.setFlags(item_dur.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 3, item_dur)
            else:
                # Incomplete event
                item_end = QTableWidgetItem("...")
                item_end.setFlags(item_end.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 2, item_end)
                
                item_dur = QTableWidgetItem("open")
                item_dur.setFlags(item_dur.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 3, item_dur)
                
                # Highlight incomplete events in yellow
                for col in range(4):
                    item = self.table.item(row, col)
                    if item:
                        item.setBackground(QBrush(QColor(255, 255, 0, 100)))
        
        # Force update and repaint
        self.table.update()
        self.table.repaint()
        self.update()
        self.repaint()
        QApplication.processEvents()

# Create widget
events_widget = EventsWidget(viewer)

# Add to viewer
try:
    dock_widget = viewer.window.add_dock_widget(
        events_widget, 
        name="Events", 
        area="right"
    )
except Exception as e:
    print(f"Error adding dock widget: {e}")
    events_widget.show()

# Initialize with empty table  
events_widget.update_table([], [])

# Ensure widget is visible after napari initializes
def ensure_visible():
    try:
        events_widget.setVisible(True)
        events_widget.show()
        QApplication.processEvents()
    except:
        pass

QTimer.singleShot(200, ensure_visible)

# this writes events and updates widget each time you press a key
def on_keypress(key, viewer):
    global open_rearing
    
    action = KEYMAP[key]
    frame = int(viewer.dims.current_step[0])

    # prevent starting new event if one is already open
    if action == 'rearing_start' and open_rearing:
        show_info('Close current event before starting new one')
        return
    
    # prevent ending if no event is open
    if action == 'rearing_end' and not open_rearing:
        show_info('No open event to close')
        return

    show_info(action)  # visual feedback

    # update state
    if action == 'rearing_start':
        open_rearing = True
    elif action == 'rearing_end':
        open_rearing = False

    # Track event
    events_coords.append(frame)
    events_actions.append(action)
    
    # Update widget
    events_widget.update_table(events_coords, events_actions)
    
    # Write CSV when event completes
    if action == 'rearing_end':
        write_csv()

def make_callback(key):
    def _cb(viewer):
        on_keypress(key, viewer)
    return _cb

def nav_left(viewer):
    """Navigate left using current step size."""
    global nav_step_size
    current = viewer.dims.current_step[0]
    new_frame = max(0, current - nav_step_size)
    viewer.dims.current_step = (new_frame, *viewer.dims.current_step[1:])

def nav_right(viewer):
    """Navigate right using current step size."""
    global nav_step_size
    current = viewer.dims.current_step[0]
    max_frame = viewer.dims.range[0][1] - 1
    new_frame = min(max_frame, current + nav_step_size)
    viewer.dims.current_step = (new_frame, *viewer.dims.current_step[1:])

def increase_nav_speed(viewer):
    """Increase navigation step size."""
    global nav_step_size
    nav_step_size = min(nav_step_size * 2, 1000)
    events_widget.nav_speed_label.setText(f"Step: {nav_step_size} frames")
    show_info(f"Navigation speed: {nav_step_size} frames")

def decrease_nav_speed(viewer):
    """Decrease navigation step size."""
    global nav_step_size
    nav_step_size = max(nav_step_size // 2, 1)
    events_widget.nav_speed_label.setText(f"Step: {nav_step_size} frames")
    show_info(f"Navigation speed: {nav_step_size} frames")

def set_playback_fps(viewer):
    """Set playback fps to 60."""
    global playback_fps
    playback_fps = 60
    events_widget.fps_btn.setText(f"Set Playback: {playback_fps} fps")
    
    # Set fps using napari's dims play
    try:
        # If playback is running, restart with new fps
        if viewer.dims.is_playing:
            # Get current axis
            axis = viewer.dims.current_step[0] if hasattr(viewer.dims, 'current_step') else 0
            viewer.dims.stop()
            viewer.dims.play(axis=0, fps=playback_fps)
        # Also set on video layers if available
        for layer in viewer.layers:
            if hasattr(layer, 'fps'):
                layer.fps = playback_fps
            elif hasattr(layer, 'play_rate'):
                layer.play_rate = playback_fps / 30.0  # Normalize if needed
        show_info(f"Playback set to {playback_fps} fps")
    except Exception as e:
        show_info(f"Setting {playback_fps} fps for next playback")
        # Store fps for when playback starts
        pass

# Bind annotation keys
for key in KEYMAP:
    viewer.bind_key(key, make_callback(key))

# Bind navigation keys
viewer.bind_key('Left', nav_left, overwrite=True)
viewer.bind_key('Right', nav_right, overwrite=True)

# Bind speed control keys
viewer.bind_key('+', increase_nav_speed, overwrite=True)
viewer.bind_key('=', increase_nav_speed, overwrite=True)  # + key (shift-=)
viewer.bind_key('-', decrease_nav_speed, overwrite=True)

# Bind fps key
viewer.bind_key('f', set_playback_fps, overwrite=True)

# Override Space key to start playback with set fps
def play_with_fps(viewer):
    """Start playback with configured fps."""
    global playback_fps
    if viewer.dims.is_playing:
        viewer.dims.stop()
    else:
        viewer.dims.play(axis=0, fps=playback_fps)

viewer.bind_key('Space', play_with_fps, overwrite=True)

napari.run()
