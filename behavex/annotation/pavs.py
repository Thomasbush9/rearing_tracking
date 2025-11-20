from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog, QHBoxLayout, QLabel, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QShortcut, QLineEdit, QSpinBox
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QUrl, QDir, QTime
from PyQt5.QtGui import QColor, QKeySequence, QStandardItemModel, QBrush
from pathlib import Path
import csv
import sys
import yaml
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

audio_extensions = [".wav", ".mp3"]
video_extensions = [".avi", ".mp4", ".mkv"]

class Window(QMainWindow):

    def __init__(self, video_path: Path | None = None):
        super().__init__()
        self.title = "Rearing Event Annotator"
        self.video_path = video_path
        self.video_fps = 60.0  # Default FPS, will be estimated if available
        self.frame_count = 0
        self.open_rearing = False
        
        # Track rearing events (frame-based)
        self.events_coords = []  # List of frame numbers
        self.events_actions = []  # List of 'rearing_start' or 'rearing_end'
        
        # Plot zoom state
        self.plot_zoom_factor = 1.0  # Zoom factor (1.0 = full view)
        self.plot_center_frame = 0  # Center frame for zoom
        self.zoom_n_frames = 100  # Default zoom window (N frames)
        
        self.InitWindow()

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self.UiComponents()
        self.show()
        
        # Auto-load video if path provided
        if self.video_path is not None:
            self.load_video_from_path(self.video_path)

    def get_current_frame(self):
        """Convert current video position (ms) to frame number."""
        if self.mediaPlayer.duration() == 0:
            return 0
        position_ms = self.mediaPlayer.position()
        frame = int((position_ms / 1000.0) * self.video_fps)
        return frame

    def get_csv_path(self):
        """Get CSV path based on current video file."""
        if not self.fileNameExist:
            return None
        video_path = Path(self.fileNameExist)
        csv_path = video_path.parent / f"{video_path.stem}_rearings.csv"
        return csv_path
    
    def check_and_load_csv(self):
        """Check if CSV exists and load it, or create empty CSV if not."""
        csv_path = self.get_csv_path()
        if csv_path is None:
            return
        
        if csv_path.exists():
            # Load existing CSV
            self.load_csv(csv_path)
            self.statusLabel.setText(f"Loaded annotations from {csv_path.name}")
            self.statusLabel.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
        else:
            # Create empty CSV with header
            self.initialize_csv(csv_path)
            self.statusLabel.setText(f"Initialized new annotation file: {csv_path.name}")
            self.statusLabel.setStyleSheet("color: blue; font-weight: bold; padding: 5px;")
            # Update session metadata if video is part of a project
            self.update_session_metadata(csv_path)
    
    def initialize_csv(self, csv_path: Path):
        """Initialize an empty CSV file with header."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'start_frame', 'end_frame', 'duration'])
    
    def load_csv(self, csv_path: Path):
        """Load events from CSV file."""
        if not csv_path.exists():
            return
        
        # Clear existing events
        self.events_coords = []
        self.events_actions = []
        self.open_rearing = False
        
        try:
            with open(csv_path, 'r') as stream:
                reader = csv.reader(stream)
                for i, row in enumerate(reader):
                    if i == 0:  # Skip header
                        continue
                    if len(row) >= 3:  # index, start_frame, end_frame, [duration]
                        try:
                            start_frame = int(row[1])
                            end_frame = int(row[2])
                            self.events_coords.append(start_frame)
                            self.events_actions.append('rearing_start')
                            self.events_coords.append(end_frame)
                            self.events_actions.append('rearing_end')
                        except (ValueError, IndexError):
                            continue
            
            # Update table and plot
            self.update_table()
        except Exception as e:
            self.errorLabel.setText(f"Error loading CSV: {e}")
            self.errorLabel.setStyleSheet('color: red')
    
    def update_session_metadata(self, csv_path: Path):
        """Update session.yaml to track annotation file if video is part of a project."""
        if not csv_path or not csv_path.exists():
            return
        
        video_path = Path(self.fileNameExist)
        if not video_path.exists():
            return
        
        # Try to find project directory by looking for sessions.yaml
        # Search parent directories for sessions.yaml
        current_dir = video_path.parent
        max_depth = 10
        depth = 0
        
        while depth < max_depth and current_dir != current_dir.parent:
            sessions_yaml = current_dir / "sessions.yaml"
            if sessions_yaml.exists():
                try:
                    # Found a project directory
                    project_dir = current_dir
                    sessions_path = sessions_yaml
                    
                    # Load sessions.yaml
                    with open(sessions_path, 'r') as f:
                        data = yaml.safe_load(f) or {}
                    
                    sessions = data.get('sessions', [])
                    
                    # Find session that contains this video
                    video_path_str = str(video_path.resolve())
                    for session in sessions:
                        views = session.get('views', {})
                        # Check if this video is in the session's views
                        if video_path_str in views.values():
                            # Update session metadata to track annotation
                            if 'annotations' not in session:
                                session['annotations'] = {}
                            
                            view_name = None
                            for vname, vpath in views.items():
                                if str(Path(vpath).resolve()) == video_path_str:
                                    view_name = vname
                                    break
                            
                            if view_name:
                                annotation_file = f"{video_path.stem}_rearings.csv"
                                if 'annotation_files' not in session['annotations']:
                                    session['annotations']['annotation_files'] = {}
                                session['annotations']['annotation_files'][view_name] = annotation_file
                                
                                # Save updated sessions.yaml
                                with open(sessions_path, 'w') as f:
                                    yaml.safe_dump({'sessions': sessions}, f, sort_keys=False, default_flow_style=False)
                                
                                print(f"Updated session.yaml: tracked annotation for {view_name}")
                            break
                except Exception as e:
                    print(f"Error updating session metadata: {e}")
                    # Don't fail if we can't update session metadata
                break
            
            current_dir = current_dir.parent
            depth += 1

    def write_csv(self):
        """Write all complete events to CSV file."""
        csv_path = self.get_csv_path()
        if csv_path is None:
            return

        # Group events into complete pairs
        events = []
        i = 0
        while i < len(self.events_coords):
            if i < len(self.events_actions) and self.events_actions[i] == 'rearing_start':
                start_frame = int(self.events_coords[i])
                # Find corresponding end
                end_frame = None
                for j in range(i + 1, len(self.events_coords)):
                    if j < len(self.events_actions) and self.events_actions[j] == 'rearing_end':
                        end_frame = int(self.events_coords[j])
                        break
                if end_frame is not None:
                    duration = end_frame - start_frame
                    events.append((start_frame, end_frame, duration))
                i += 1
            else:
                i += 1

        # Write complete CSV file
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_is_new = not csv_path.exists()
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'start_frame', 'end_frame', 'duration'])
            for idx, (start, end, duration) in enumerate(events, 1):
                writer.writerow([idx, start, end, duration])
        
        # Update session metadata if this is a new file
        if file_is_new:
            self.update_session_metadata(csv_path)

    def load_video_from_path(self, video_path: Path):
        """Load video from given path."""
        video_path = Path(video_path).expanduser().resolve()
        if not video_path.exists():
            self.errorLabel.setText(f"Video file not found: {video_path}")
            self.errorLabel.setStyleSheet('color: red')
            return
        
        self.fileNameExist = str(video_path)
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.fileNameExist)))
        self.playButton.setEnabled(True)
        self.videopath = QUrl.fromLocalFile(self.fileNameExist)
        self.errorLabel.setText(self.fileNameExist)
        self.errorLabel.setStyleSheet('color: black')
        
        # Check for existing CSV and load it, or initialize new one
        self.check_and_load_csv()

    def UiComponents(self):

        self.rowNo = 1
        self.colNo = 0
        self.fName = ""
        self.fName2 = ""
        self.fileNameExist = ""
        self.dropDownName = ""

        self.model = QStandardItemModel()

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.tableWidget = QTableWidget()
        self.tableWidget.cellClicked.connect(self.checkTableFrame)

        self.videoWidget = QVideoWidget()
        self.frameID = 0

        self.insertBaseRow()

        openButton = QPushButton("Open...")
        openButton.clicked.connect(self.openFile)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        # Time label (for display)
        self.lbl = QLabel('00:00:00')
        self.lbl.setFixedWidth(60)
        self.lbl.setUpdatesEnabled(True)

        # Frame label
        self.frameLabel = QLabel('Frame: 0')
        self.frameLabel.setFixedWidth(80)
        self.frameLabel.setUpdatesEnabled(True)

        # Duration label
        self.elbl = QLabel('00:00:00')
        self.elbl.setFixedWidth(60)
        self.elbl.setUpdatesEnabled(True)

        self.delButton = QPushButton("Delete")
        self.delButton.clicked.connect(self.delete)

        self.exportButton = QPushButton("Export")
        self.exportButton.clicked.connect(self.export)

        self.importButton = QPushButton("Import")
        self.importButton.clicked.connect(self.importCSV)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 100)
        self.positionSlider.sliderMoved.connect(self.setPosition)
        self.positionSlider.sliderMoved.connect(self.handleLabel)
        self.positionSlider.setSingleStep(2)
        self.positionSlider.setPageStep(20)
        self.positionSlider.setAttribute(Qt.WA_TranslucentBackground, True)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # Status label for rearing events
        self.statusLabel = QLabel("No events yet")
        self.statusLabel.setStyleSheet("color: blue; font-weight: bold; padding: 5px;")

        # Matplotlib widget for event visualization
        self.fig = Figure(figsize=(8, 2), facecolor='white')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Events')
        self.ax.set_ylim(0, 1)
        self.canvas.setMinimumHeight(150)
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        # Zoom controls
        zoomLabel = QLabel("Zoom:")
        self.zoomSpinBox = QSpinBox()
        self.zoomSpinBox.setMinimum(10)
        self.zoomSpinBox.setMaximum(10000)
        self.zoomSpinBox.setValue(self.zoom_n_frames)
        self.zoomSpinBox.setSuffix(" frames")
        self.zoomSpinBox.valueChanged.connect(self.on_zoom_changed)
        
        zoomInButton = QPushButton("Zoom In [+]")
        zoomInButton.clicked.connect(self.zoom_in)
        
        zoomOutButton = QPushButton("Zoom Out [-]")
        zoomOutButton.clicked.connect(self.zoom_out)
        
        resetZoomButton = QPushButton("Reset Zoom")
        resetZoomButton.clicked.connect(self.reset_zoom)
        
        zoomControls = QHBoxLayout()
        zoomControls.addWidget(zoomLabel)
        zoomControls.addWidget(self.zoomSpinBox)
        zoomControls.addWidget(zoomInButton)
        zoomControls.addWidget(zoomOutButton)
        zoomControls.addWidget(resetZoomButton)

        # Main plotBox
        plotBox = QHBoxLayout()

        controlLayout = QHBoxLayout()
        controlLayout.addWidget(openButton)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.lbl)
        controlLayout.addWidget(self.frameLabel)
        controlLayout.addWidget(self.positionSlider)
        controlLayout.addWidget(self.elbl)

        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Left Layout
        layout = QVBoxLayout()
        layout.addWidget(self.videoWidget, 3)
        layout.addLayout(controlLayout)
        layout.addWidget(self.statusLabel)
        layout.addWidget(self.errorLabel)
        
        # Add matplotlib plot
        plotLayout = QVBoxLayout()
        plotLayout.addLayout(zoomControls)
        plotLayout.addWidget(self.canvas)
        layout.addLayout(plotLayout, 1)

        plotBox.addLayout(layout, 5)

        # Right Layout
        feats = QHBoxLayout()
        feats.addWidget(self.delButton)
        feats.addWidget(self.exportButton)
        feats.addWidget(self.importButton)

        layout2 = QVBoxLayout()
        layout2.addWidget(self.tableWidget)
        layout2.addLayout(feats, 1)

        plotBox.addLayout(layout2, 2)

        wid.setLayout(plotBox)

        # Keyboard shortcuts: h for rearing_start, j for rearing_end
        self.shortcut = QShortcut(QKeySequence("h"), self)
        self.shortcut.activated.connect(self.add_rearing_start)
        self.shortcut = QShortcut(QKeySequence("j"), self)
        self.shortcut.activated.connect(self.add_rearing_end)
        self.shortcut = QShortcut(QKeySequence("L"), self)
        self.shortcut.activated.connect(self.openFile)
        self.shortcut = QShortcut(QKeySequence("C"), self)
        self.shortcut.activated.connect(self.clearTable)
        self.shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut.activated.connect(self.play)

        self.shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut.activated.connect(self.forwardSlider)
        self.shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut.activated.connect(self.backSlider)
        self.shortcut = QShortcut(QKeySequence(Qt.Key_Up), self)
        self.shortcut.activated.connect(self.volumeUp)
        self.shortcut = QShortcut(QKeySequence(Qt.Key_Down), self)
        self.shortcut.activated.connect(self.volumeDown)
        self.shortcut = QShortcut(QKeySequence(Qt.ShiftModifier + Qt.Key_Right), self)
        self.shortcut.activated.connect(self.forwardSlider10)
        self.shortcut = QShortcut(QKeySequence(Qt.ShiftModifier + Qt.Key_Left), self)
        self.shortcut.activated.connect(self.backSlider10)

        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.positionChanged.connect(self.handleLabel)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        
        # Initialize plot
        self.update_plot()

    def add_rearing_start(self):
        """Mark rearing start event at current frame."""
        if self.open_rearing:
            self.statusLabel.setText("Close current event before starting new one")
            self.statusLabel.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
            return
        
        frame = self.get_current_frame()
        self.events_coords.append(frame)
        self.events_actions.append('rearing_start')
        self.open_rearing = True
        self.update_table()  # This will call update_plot()
        self.statusLabel.setText(f"Rearing start at frame {frame}")
        self.statusLabel.setStyleSheet("color: green; font-weight: bold; padding: 5px;")

    def add_rearing_end(self):
        """Mark rearing end event at current frame."""
        if not self.open_rearing:
            self.statusLabel.setText("No open event to close")
            self.statusLabel.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
            return
        
        frame = self.get_current_frame()
        self.events_coords.append(frame)
        self.events_actions.append('rearing_end')
        self.open_rearing = False
        self.update_table()  # This will call update_plot()
        self.write_csv()  # Auto-save when event completes
        self.statusLabel.setText(f"Rearing end at frame {frame} - Saved!")
        self.statusLabel.setStyleSheet("color: blue; font-weight: bold; padding: 5px;")

    def update_table(self):
        """Update the table with current rearing events."""
        # Group start/end pairs
        events = []
        i = 0
        while i < len(self.events_coords):
            if i < len(self.events_actions) and self.events_actions[i] == 'rearing_start':
                start_frame = int(self.events_coords[i])
                # Find corresponding end
                end_frame = None
                for j in range(i + 1, len(self.events_coords)):
                    if j < len(self.events_actions) and self.events_actions[j] == 'rearing_end':
                        end_frame = int(self.events_coords[j])
                        break
                if end_frame is not None:
                    duration = end_frame - start_frame
                    events.append((start_frame, end_frame, duration))
                else:
                    # Incomplete event
                    events.append((start_frame, None, None))
                i += 1
            else:
                i += 1

        # Update status
        if len(events) == 0:
            complete = 0
        else:
            complete = sum(1 for e in events if e[1] is not None)
        
        if len(events) > 0:
            status_text = f"{len(events)} event(s) - {complete} complete"
            if self.open_rearing:
                status_text += " [Event open]"
        else:
            status_text = "No events yet"
        
        # Clear and repopulate table
        self.tableWidget.setRowCount(0)
        self.tableWidget.setRowCount(len(events))
        
        for row, (start, end, duration) in enumerate(events):
            # Index
            item_idx = QTableWidgetItem(str(row + 1))
            item_idx.setFlags(item_idx.flags() & ~Qt.ItemIsEditable)
            self.tableWidget.setItem(row, 0, item_idx)
            
            # Start frame
            item_start = QTableWidgetItem(str(start))
            item_start.setFlags(item_start.flags() & ~Qt.ItemIsEditable)
            self.tableWidget.setItem(row, 1, item_start)
            
            if end is not None:
                # End frame
                item_end = QTableWidgetItem(str(end))
                item_end.setFlags(item_end.flags() & ~Qt.ItemIsEditable)
                self.tableWidget.setItem(row, 2, item_end)
                
                # Duration
                item_dur = QTableWidgetItem(f"{duration} frames")
                item_dur.setFlags(item_dur.flags() & ~Qt.ItemIsEditable)
                self.tableWidget.setItem(row, 3, item_dur)
            else:
                # Incomplete event
                item_end = QTableWidgetItem("...")
                item_end.setFlags(item_end.flags() & ~Qt.ItemIsEditable)
                self.tableWidget.setItem(row, 2, item_end)
                
                item_dur = QTableWidgetItem("open")
                item_dur.setFlags(item_dur.flags() & ~Qt.ItemIsEditable)
                self.tableWidget.setItem(row, 3, item_dur)
                
                # Highlight incomplete events
                for col in range(4):
                    item = self.tableWidget.item(row, col)
                    if item:
                        item.setBackground(QBrush(QColor(255, 255, 0, 100)))
        
        # Update plot after table update
        self.update_plot()

    def update_plot(self):
        """Update the matplotlib plot with current events."""
        self.ax.clear()
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Events')
        self.ax.set_ylim(0, 1)
        
        # Get complete events (start/end pairs)
        events = []
        i = 0
        while i < len(self.events_coords):
            if i < len(self.events_actions) and self.events_actions[i] == 'rearing_start':
                start_frame = int(self.events_coords[i])
                end_frame = None
                for j in range(i + 1, len(self.events_coords)):
                    if j < len(self.events_actions) and self.events_actions[j] == 'rearing_end':
                        end_frame = int(self.events_coords[j])
                        break
                if end_frame is not None:
                    events.append((start_frame, end_frame))
                i += 1
            else:
                i += 1
        
        # Draw event bars (red with alpha 0.3)
        for start, end in events:
            width = end - start
            rect = Rectangle((start, 0), width, 1, facecolor='red', alpha=0.3, edgecolor='darkred', linewidth=0.5)
            self.ax.add_patch(rect)
        
        # Get current frame for indicator
        current_frame = self.get_current_frame()
        
        # Determine x-axis limits based on zoom
        if self.plot_zoom_factor == 1.0:
            # Full view: show all frames
            if self.frame_count > 0:
                x_min, x_max = 0, self.frame_count
            elif events:
                max_frame = max(end for _, end in events)
                x_min, x_max = 0, max(max_frame + 100, 100)
            else:
                x_min, x_max = 0, max(current_frame + 100, 100)
        else:
            # Zoomed view: show N frames around center
            half_window = self.zoom_n_frames // 2
            x_min = max(0, self.plot_center_frame - half_window)
            x_max = self.plot_center_frame + half_window
        
        self.ax.set_xlim(x_min, x_max)
        
        # Draw current frame indicator (vertical line)
        if x_min <= current_frame <= x_max:
            self.ax.axvline(x=current_frame, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Current frame')
        
        # Draw zoom center indicator if zoomed
        if self.plot_zoom_factor != 1.0 and x_min <= self.plot_center_frame <= x_max:
            self.ax.axvline(x=self.plot_center_frame, color='green', linestyle=':', linewidth=1, alpha=0.5)
        
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f'Rearing Events (Zoom: {self.zoom_n_frames} frames, Center: {self.plot_center_frame})')
        self.canvas.draw()

    def on_plot_click(self, event):
        """Handle clicks on the plot to jump to frame."""
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # Left click
            clicked_frame = int(event.xdata)
            # Convert frame to position in milliseconds
            position_ms = int((clicked_frame / self.video_fps) * 1000)
            self.mediaPlayer.setPosition(position_ms)
            # Update zoom center to clicked frame
            self.plot_center_frame = clicked_frame
            self.update_plot()

    def zoom_in(self):
        """Zoom in around current frame."""
        current_frame = self.get_current_frame()
        self.plot_center_frame = current_frame
        self.zoom_n_frames = max(10, self.zoom_n_frames // 2)
        self.zoomSpinBox.setValue(self.zoom_n_frames)
        self.plot_zoom_factor = 0.5
        self.update_plot()

    def zoom_out(self):
        """Zoom out around current frame."""
        current_frame = self.get_current_frame()
        self.plot_center_frame = current_frame
        self.zoom_n_frames = min(10000, self.zoom_n_frames * 2)
        self.zoomSpinBox.setValue(self.zoom_n_frames)
        if self.zoom_n_frames >= self.frame_count and self.frame_count > 0:
            self.plot_zoom_factor = 1.0
        self.update_plot()

    def reset_zoom(self):
        """Reset zoom to full view."""
        self.plot_zoom_factor = 1.0
        if self.frame_count > 0:
            self.zoom_n_frames = self.frame_count
        else:
            self.zoom_n_frames = 100
        self.zoomSpinBox.setValue(self.zoom_n_frames)
        self.update_plot()

    def on_zoom_changed(self, value):
        """Handle zoom spinbox value change."""
        self.zoom_n_frames = value
        current_frame = self.get_current_frame()
        self.plot_center_frame = current_frame
        if self.frame_count > 0 and value >= self.frame_count:
            self.plot_zoom_factor = 1.0
        else:
            self.plot_zoom_factor = 0.5
        self.update_plot()

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie", QDir.homePath())

        if fileName != '':
            self.fileNameExist = fileName
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.videopath = QUrl.fromLocalFile(fileName)
            self.errorLabel.setText(fileName)
            self.errorLabel.setStyleSheet('color: black')
            # Check for existing CSV and load it, or initialize new one
            self.check_and_load_csv()

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def delete(self):
        index_list = []
        for model_index in self.tableWidget.selectionModel().selectedRows():
            index = QtCore.QPersistentModelIndex(model_index)
            index_list.append(index)

        # Rebuild events list without deleted items
        # This is simplified - in practice you'd need to track which events correspond to which rows
        self.update_table()

    def clearTable(self):
        self.events_coords = []
        self.events_actions = []
        self.open_rearing = False
        self.update_table()
        print("Cleared all events")

    def export(self):
        """Export current events to CSV."""
        self.write_csv()
        csv_path = self.get_csv_path()
        if csv_path:
            self.statusLabel.setText(f"Exported to {csv_path.name}")
            self.statusLabel.setStyleSheet("color: green; font-weight: bold; padding: 5px;")

    def importCSV(self):
        self.clearTable()
        path, _ = QFileDialog.getOpenFileName(self, 'Import CSV', QDir.homePath(), "CSV Files(*.csv *.txt)")
        print(path)
        if path:
            with open(path, 'r') as stream:
                print("loading", path)
                reader = csv.reader(stream)
                for i, row in enumerate(reader):
                    if i == 0:  # Skip header
                        continue
                    if len(row) >= 3:  # index, start_frame, end_frame, [duration]
                        try:
                            start_frame = int(row[1])
                            end_frame = int(row[2])
                            self.events_coords.append(start_frame)
                            self.events_actions.append('rearing_start')
                            self.events_coords.append(end_frame)
                            self.events_actions.append('rearing_end')
                        except ValueError:
                            continue
                self.update_table()

    def insertBaseRow(self):
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setHorizontalHeaderLabels(["Index", "Start Frame", "End Frame", "Duration"])
        self.tableWidget.setRowCount(0)
        self.rowNo = 1
        self.colNo = 0

    def checkTableFrame(self, row, column):
        """Jump to frame when table cell is clicked."""
        if row >= 0 and column in [1, 2]:
            item = self.tableWidget.item(row, column)
            if item and item.text() and item.text() not in ["...", ""]:
                try:
                    frame = int(item.text())
                    # Convert frame to position in milliseconds
                    position_ms = int((frame / self.video_fps) * 1000)
                    self.mediaPlayer.setPosition(position_ms)
                except (ValueError, ZeroDivisionError):
                    self.errorLabel.setText("Error: Invalid frame number")
                    self.errorLabel.setStyleSheet('color: red')

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        """Update slider range and estimate FPS if possible."""
        self.positionSlider.setRange(0, duration)
        mtime = QTime(0, 0, 0, 0)
        mtime = mtime.addMSecs(self.mediaPlayer.duration())
        self.elbl.setText(mtime.toString())
        
        # Estimate frame count and FPS (simplified - using default 60 fps)
        # In practice, you might want to use OpenCV or other libraries to get exact FPS
        if duration > 0:
            duration_sec = duration / 1000.0
            # Assume 60 fps if not known (could be improved with video metadata)
            if self.frame_count == 0:
                self.frame_count = int(duration_sec * self.video_fps)
                # Update plot with new frame count
                self.update_plot()

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())
        self.errorLabel.setStyleSheet('color: red')

    def forwardSlider(self):
        """Move forward by a few frames."""
        frame_step_ms = int((1 / self.video_fps) * 1000)  # 1 frame in ms
        self.mediaPlayer.setPosition(self.mediaPlayer.position() + frame_step_ms)

    def forwardSlider10(self):
        """Move forward by many frames."""
        frame_step_ms = int((10 / self.video_fps) * 1000)  # 10 frames in ms
        self.mediaPlayer.setPosition(self.mediaPlayer.position() + frame_step_ms)

    def backSlider(self):
        """Move backward by a few frames."""
        frame_step_ms = int((1 / self.video_fps) * 1000)  # 1 frame in ms
        self.mediaPlayer.setPosition(max(0, self.mediaPlayer.position() - frame_step_ms))

    def backSlider10(self):
        """Move backward by many frames."""
        frame_step_ms = int((10 / self.video_fps) * 1000)  # 10 frames in ms
        self.mediaPlayer.setPosition(max(0, self.mediaPlayer.position() - frame_step_ms))

    def volumeUp(self):
        self.mediaPlayer.setVolume(self.mediaPlayer.volume() + 10)
        print("Volume: " + str(self.mediaPlayer.volume()))

    def volumeDown(self):
        self.mediaPlayer.setVolume(self.mediaPlayer.volume() - 10)
        print("Volume: " + str(self.mediaPlayer.volume()))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def handleLabel(self):
        """Update time and frame labels."""
        mtime = QTime(0, 0, 0, 0)
        self.time = mtime.addMSecs(self.mediaPlayer.position())
        self.lbl.setText(self.time.toString())
        
        # Update frame label
        frame = self.get_current_frame()
        self.frameLabel.setText(f"Frame: {frame}")
        
        # Update plot to show current frame indicator (throttle updates for performance)
        # Only update if zoomed (not full view) to reduce overhead
        if self.plot_zoom_factor != 1.0:
            if hasattr(self, '_last_plot_update_frame'):
                if abs(frame - self._last_plot_update_frame) > 10:  # Update every 10 frames when zoomed
                    self.update_plot()
                    self._last_plot_update_frame = frame
            else:
                self._last_plot_update_frame = frame
                self.update_plot()

    def dropEvent(self, event):
        f = str(event.mimeData().urls()[0].toLocalFile())
        self.load_video_from_path(Path(f))


def start_app(video_path: Path | None = None):
    """
    Create and run the rearing event annotator.
    
    Args:
        video_path: Optional path to video file to auto-load
    
    Usage:
        from behavex.annotation.pavs import start_app
        start_app(Path("/path/to/video.mp4"))
    """
    app = QApplication(sys.argv)
    window = Window(video_path=video_path)
    sys.exit(app.exec_())


if __name__ == "__main__":
    video_path_arg = None
    if len(sys.argv) > 1:
        video_path_arg = Path(sys.argv[1])
    
    start_app(video_path_arg)
