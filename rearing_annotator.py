import csv
from pathlib import Path

import napari
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QLabel, QApplication
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

for key in KEYMAP:
    viewer.bind_key(key, make_callback(key))

napari.run()
