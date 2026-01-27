"""The Code has been adapted from the following source:
https://github.com/kevalvc/Python-Annotator-for-VideoS
"""

from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog, QHBoxLayout, QLabel, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QShortcut, QLineEdit, QSpinBox, QDialog, QListWidget, QListWidgetItem, QCheckBox, QScrollArea, QMenu, QAction, QComboBox, QStatusBar
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QUrl, QDir, QTime
from PyQt5.QtGui import QColor, QKeySequence, QStandardItemModel, QBrush
from pathlib import Path
import csv
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

audio_extensions = [".wav", ".mp3"]
video_extensions = [".avi", ".mp4", ".mkv"]

class Window(QMainWindow):

    def __init__(self, video_path: Path | None = None, project=None, sessions=None):
        super().__init__()
        self.title = "Rearing Event Annotator"
        self.video_path = video_path
        self.project = project  # Store project reference for session switching
        self.sessions = sessions  # Store sessions list for switching
        self.video_fps = 60.0  # Default FPS, will be estimated if available
        self.frame_count = 0
        self.open_rearing = False
        self.current_behavior = "rearing"  # Default behavior, can be changed in GUI

        # Track rearing events (frame-based)
        self.events_coords = []  # List of frame numbers
        self.events_actions = []  # List of 'rearing_start' or 'rearing_end'
        self.events_labels = []  # List of behavior labels for each event pair

        # Color palette for behavior classes (colorblind-friendly)
        self.behavior_colors = {
            "rearing": ("#E69F00", "#CC8400"),      # Orange
            "grooming": ("#56B4E9", "#3A9BD9"),     # Sky blue
            "feeding": ("#009E73", "#007A59"),      # Teal
            "drinking": ("#F0E442", "#D4C700"),     # Yellow
            "exploring": ("#0072B2", "#005A8C"),    # Blue
            "resting": ("#D55E00", "#B34D00"),      # Vermillion
            "other": ("#CC79A7", "#A6628A"),        # Pink
        }
        self.default_color = ("#999999", "#666666")  # Gray for unknown behaviors
        
        # Plot zoom state
        self.plot_zoom_factor = 1.0  # Zoom factor (1.0 = full view)
        self.plot_center_frame = 0  # Center frame for zoom
        self.zoom_n_frames = 100  # Default zoom window (N frames)
        
        # Persistent plot objects (for efficient updates without clearing)
        self.event_rectangles = []  # List of Rectangle patches for events
        self.frame_indicator_line = None  # Line2D for current frame indicator
        self.zoom_indicator_line = None  # Line2D for zoom center indicator
        
        # Feature data
        self.features_df = None  # DataFrame with features
        self.selected_features = []  # List of selected feature names to plot
        self.current_session = None  # Current session object for feature import
        self.feature_checkboxes = {}  # Dictionary of feature checkboxes
        
        # Feature plot (separate window)
        self.feature_plot_window = None  # Separate window for feature plots
        self.feature_fig = None  # Matplotlib figure for features
        self.feature_canvas = None  # Canvas for feature plot
        self.feature_ax = None  # Axis for feature plot
        self._last_feature_plot_update_frame = None  # Track last feature plot update
        # Persistent plot objects for efficient updates
        self.feature_frame_indicators = []  # Persistent Line2D objects for frame indicators (one per subplot)
        self.feature_lines = []  # Persistent Line2D objects for feature data lines
        self.feature_event_rectangles = []  # List of Rectangle patches per subplot
        self.feature_axes = []  # Store axes references
        self._last_drawn_events_hash = None  # Hash of last drawn events to detect changes

        # Status tracking
        self._last_save_time = None  # Track last save timestamp

        # Undo/redo history
        self._undo_stack = []  # List of (events_coords, events_actions, events_labels, open_rearing) tuples
        self._redo_stack = []
        self._max_history = 50  # Maximum undo levels

        self.InitWindow()

    def get_available_behaviors(self):
        """Get list of available behaviors from project config or return defaults.

        Behaviors can be configured per-project in config.yaml:
            annotation:
              behaviors: ["rearing", "grooming", "exploring"]

        If not configured, defaults to common rodent behaviors (without feeding).
        """
        default_behaviors = ["rearing", "grooming", "exploring", "resting", "other"]

        if self.project is not None and hasattr(self.project, 'config'):
            # Try to get behaviors from project config
            annotation_config = self.project.config.get("annotation", {})
            behaviors = annotation_config.get("behaviors", None)
            if behaviors and isinstance(behaviors, list):
                return behaviors
            # Fallback to single behavior field for backward compatibility
            single_behavior = annotation_config.get("behavior", None)
            if single_behavior:
                return [single_behavior] + [b for b in default_behaviors if b != single_behavior]
        
        return default_behaviors

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
            writer.writerow(['index', 'start_frame', 'end_frame', 'duration', 'label'])
    
    def load_csv(self, csv_path: Path):
        """Load events from CSV file."""
        if not csv_path.exists():
            return
        
        # Clear existing events
        self.events_coords = []
        self.events_actions = []
        self.events_labels = []
        self.open_rearing = False
        
        try:
            with open(csv_path, 'r') as stream:
                reader = csv.reader(stream)
                header = next(reader, None)  # Get header to check format
                has_label = header and 'label' in [col.lower() for col in header]
                
                for i, row in enumerate(reader):
                    if len(row) >= 3:  # index, start_frame, end_frame, [duration, label]
                        try:
                            start_frame = int(row[1])
                            end_frame = int(row[2])
                            # Get label if present, otherwise default to "rearing" for backward compatibility
                            label = row[4].strip() if has_label and len(row) > 4 and row[4].strip() else "rearing"
                            
                            self.events_coords.append(start_frame)
                            self.events_actions.append('rearing_start')
                            self.events_coords.append(end_frame)
                            self.events_actions.append('rearing_end')
                            self.events_labels.append(label)  # Store label for this event pair
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

        # Group events into complete pairs with labels
        events = []
        event_idx = 0
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
                    # Get label for this event (use stored label or default to current_behavior)
                    label = self.events_labels[event_idx] if event_idx < len(self.events_labels) else self.current_behavior
                    events.append((start_frame, end_frame, duration, label))
                    event_idx += 1
                i += 1
            else:
                i += 1

        # Write complete CSV file
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_is_new = not csv_path.exists()
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'start_frame', 'end_frame', 'duration', 'label'])
            for idx, (start, end, duration, label) in enumerate(events, 1):
                writer.writerow([idx, start, end, duration, label])

        # Record save time
        from datetime import datetime
        self._last_save_time = datetime.now().strftime("%H:%M:%S")

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
        
        # Find current session if project is available
        if self.project is not None and self.sessions is not None:
            video_path_str = str(video_path.resolve())
            for session in self.sessions:
                views = session.views
                if video_path_str in views.values():
                    self.current_session = session
                    break
        
        # Check for existing CSV and load it, or initialize new one
        self.check_and_load_csv()
        
        # Try to load features for this session
        self.load_features_for_session()

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
        self.tableWidget.cellChanged.connect(self._on_table_cell_edited)
        self._table_updating = False  # Flag to prevent recursive updates

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
        
        # Feature import button with context menu
        self.importFeaturesButton = QPushButton("Import Features")
        self.importFeaturesButton.clicked.connect(self.import_features)
        self.importFeaturesButton.setContextMenuPolicy(Qt.CustomContextMenu)
        self.importFeaturesButton.customContextMenuRequested.connect(self.show_feature_menu)
        
        # Behavior selector combo box
        behaviorLabel = QLabel("Behavior:")
        behaviorLabel.setStyleSheet("font-weight: bold;")
        self.behaviorComboBox = QComboBox()
        available_behaviors = self.get_available_behaviors()
        self.behaviorComboBox.addItems(available_behaviors)
        # Set current behavior if it's in the list
        if self.current_behavior in available_behaviors:
            self.behaviorComboBox.setCurrentText(self.current_behavior)
        self.behaviorComboBox.currentTextChanged.connect(self.on_behavior_changed)

        # Keyboard shortcuts help label
        shortcuts_text = "Keys: H=start, J=end, 1-9=behavior, Space=play"
        self.shortcutsLabel = QLabel(shortcuts_text)
        self.shortcutsLabel.setStyleSheet("color: gray; font-size: 10px; padding: 2px;")
        
        # Add switch session button if project/sessions available
        self.switchSessionButton = None
        if self.project is not None and self.sessions is not None:
            self.switchSessionButton = QPushButton("Switch Session")
            self.switchSessionButton.clicked.connect(self.switch_session)

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
        self.fig = Figure(figsize=(8, 6), facecolor='white')
        self.canvas = FigureCanvas(self.fig)
        self.axes = []  # List of axes (will be created in update_plot)
        self.ax = None  # Main axis for click handling
        self.canvas.setMinimumHeight(200)
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        # Feature selection widget
        self.featureSelectionWidget = QWidget()
        featureSelectionLayout = QVBoxLayout()
        featureSelectionLayout.setContentsMargins(5, 5, 5, 5)
        
        featureLabel = QLabel("Features to Plot:")
        featureLabel.setStyleSheet("font-weight: bold;")
        featureSelectionLayout.addWidget(featureLabel)
        
        # Scroll area for feature checkboxes
        self.featureScrollArea = QScrollArea()
        self.featureScrollArea.setWidgetResizable(True)
        self.featureScrollArea.setMaximumHeight(150)
        self.featureCheckboxWidget = QWidget()
        self.featureCheckboxLayout = QVBoxLayout()
        self.featureCheckboxWidget.setLayout(self.featureCheckboxLayout)
        self.featureScrollArea.setWidget(self.featureCheckboxWidget)
        featureSelectionLayout.addWidget(self.featureScrollArea)
        
        # Button to open feature plot window
        self.viewFeaturesButton = QPushButton("Open Feature Plot")
        self.viewFeaturesButton.clicked.connect(self.open_feature_plot_window)
        self.viewFeaturesButton.setVisible(False)  # Hidden until features are loaded
        featureSelectionLayout.addWidget(self.viewFeaturesButton)
        
        self.featureSelectionWidget.setLayout(featureSelectionLayout)
        
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
        # Behavior selector
        behaviorLayout = QHBoxLayout()
        behaviorLayout.addWidget(behaviorLabel)
        behaviorLayout.addWidget(self.behaviorComboBox)

        # Shortcuts help
        shortcutsLayout = QHBoxLayout()
        shortcutsLayout.addWidget(self.shortcutsLabel)

        feats = QHBoxLayout()
        feats.addWidget(self.delButton)
        feats.addWidget(self.exportButton)
        feats.addWidget(self.importButton)
        feats.addWidget(self.importFeaturesButton)
        if self.switchSessionButton is not None:
            feats.addWidget(self.switchSessionButton)

        layout2 = QVBoxLayout()
        layout2.addLayout(behaviorLayout)
        layout2.addLayout(shortcutsLayout)
        layout2.addWidget(self.tableWidget)
        layout2.addLayout(feats, 1)

        # Add feature selection widget
        layout2.addWidget(self.featureSelectionWidget)

        plotBox.addLayout(layout2, 2)

        wid.setLayout(plotBox)

        # Keyboard shortcuts: h for rearing_start, j for rearing_end
        self.shortcut_start = QShortcut(QKeySequence("h"), self)
        self.shortcut_start.activated.connect(self.add_rearing_start)
        self.shortcut_end = QShortcut(QKeySequence("j"), self)
        self.shortcut_end.activated.connect(self.add_rearing_end)
        self.shortcut_open = QShortcut(QKeySequence("L"), self)
        self.shortcut_open.activated.connect(self.openFile)
        self.shortcut_clear = QShortcut(QKeySequence("C"), self)
        self.shortcut_clear.activated.connect(self.clearTable)
        self.shortcut_play = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_play.activated.connect(self.play)

        # Undo/Redo shortcuts
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo)
        self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self.shortcut_redo.activated.connect(self.redo)

        # Number keys 1-9 for quick behavior selection
        self.behavior_shortcuts = []
        for i in range(1, 10):
            shortcut = QShortcut(QKeySequence(str(i)), self)
            shortcut.activated.connect(lambda idx=i-1: self.select_behavior_by_index(idx))
            self.behavior_shortcuts.append(shortcut)

        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(self.forwardSlider)
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(self.backSlider)
        self.shortcut_up = QShortcut(QKeySequence(Qt.Key_Up), self)
        self.shortcut_up.activated.connect(self.volumeUp)
        self.shortcut_down = QShortcut(QKeySequence(Qt.Key_Down), self)
        self.shortcut_down.activated.connect(self.volumeDown)
        self.shortcut_right10 = QShortcut(QKeySequence(Qt.ShiftModifier + Qt.Key_Right), self)
        self.shortcut_right10.activated.connect(self.forwardSlider10)
        self.shortcut_left10 = QShortcut(QKeySequence(Qt.ShiftModifier + Qt.Key_Left), self)
        self.shortcut_left10.activated.connect(self.backSlider10)

        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.positionChanged.connect(self.handleLabel)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        
        # Initialize plot (will create empty plot initially)
        # Initialize with empty axes
        self.ax = self.fig.add_subplot(111)
        self.axes = [self.ax]
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Events')
        self.ax.set_ylim(0, 1)
        self.canvas.draw()

        # Status bar at bottom of window
        self._init_status_bar()

    def _init_status_bar(self):
        """Initialize the status bar with persistent labels."""
        self.main_status_bar = QStatusBar()
        self.setStatusBar(self.main_status_bar)

        # Behavior indicator
        self.sb_behavior = QLabel()
        self.sb_behavior.setStyleSheet("padding: 2px 8px; font-weight: bold;")
        self.main_status_bar.addWidget(self.sb_behavior)

        # Event state indicator
        self.sb_event_state = QLabel()
        self.sb_event_state.setStyleSheet("padding: 2px 8px;")
        self.main_status_bar.addWidget(self.sb_event_state)

        # Event count
        self.sb_event_count = QLabel()
        self.sb_event_count.setStyleSheet("padding: 2px 8px;")
        self.main_status_bar.addWidget(self.sb_event_count)

        # Last save time (right-aligned)
        self.sb_save_time = QLabel()
        self.sb_save_time.setStyleSheet("padding: 2px 8px; color: gray;")
        self.main_status_bar.addPermanentWidget(self.sb_save_time)

        self._update_status_bar()

    def _save_undo_state(self):
        """Save current state to undo stack before modifying events."""
        state = (
            self.events_coords.copy(),
            self.events_actions.copy(),
            self.events_labels.copy(),
            self.open_rearing
        )
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()  # Clear redo on new action

    def _restore_state(self, state):
        """Restore events from a saved state tuple."""
        self.events_coords, self.events_actions, self.events_labels, self.open_rearing = (
            state[0].copy(), state[1].copy(), state[2].copy(), state[3]
        )
        self.update_table()
        self._update_status_bar()

    def undo(self):
        """Undo last event action."""
        if not self._undo_stack:
            self.statusLabel.setText("Nothing to undo")
            self.statusLabel.setStyleSheet("color: gray; font-weight: bold; padding: 5px;")
            return

        # Save current state to redo stack
        current = (
            self.events_coords.copy(),
            self.events_actions.copy(),
            self.events_labels.copy(),
            self.open_rearing
        )
        self._redo_stack.append(current)

        # Restore previous state
        state = self._undo_stack.pop()
        self._restore_state(state)
        self.write_csv()
        self.statusLabel.setText("Undo")
        self.statusLabel.setStyleSheet("color: blue; font-weight: bold; padding: 5px;")

    def redo(self):
        """Redo previously undone action."""
        if not self._redo_stack:
            self.statusLabel.setText("Nothing to redo")
            self.statusLabel.setStyleSheet("color: gray; font-weight: bold; padding: 5px;")
            return

        # Save current state to undo stack
        current = (
            self.events_coords.copy(),
            self.events_actions.copy(),
            self.events_labels.copy(),
            self.open_rearing
        )
        self._undo_stack.append(current)

        # Restore redo state
        state = self._redo_stack.pop()
        self._restore_state(state)
        self.write_csv()
        self.statusLabel.setText("Redo")
        self.statusLabel.setStyleSheet("color: blue; font-weight: bold; padding: 5px;")

    def _update_status_bar(self):
        # Behavior
        color = self.behavior_colors.get(self.current_behavior, self.default_color)[0]
        self.sb_behavior.setText(f"Behavior: {self.current_behavior.upper()}")
        self.sb_behavior.setStyleSheet(f"padding: 2px 8px; font-weight: bold; color: {color};")

        # Event state
        if self.open_rearing:
            self.sb_event_state.setText("EVENT OPEN")
            self.sb_event_state.setStyleSheet("padding: 2px 8px; background: #ffeb3b; color: black; font-weight: bold;")
        else:
            self.sb_event_state.setText("Ready")
            self.sb_event_state.setStyleSheet("padding: 2px 8px; color: gray;")

        # Event count
        complete_events = len(self.events_labels)
        self.sb_event_count.setText(f"Events: {complete_events}")

        # Last save time
        if self._last_save_time:
            self.sb_save_time.setText(f"Saved: {self._last_save_time}")
        else:
            self.sb_save_time.setText("Not saved")

    def add_rearing_start(self):
        """Mark rearing start event at current frame."""
        if self.open_rearing:
            self.statusLabel.setText("Close current event before starting new one")
            self.statusLabel.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
            return

        self._save_undo_state()
        frame = self.get_current_frame()
        self.events_coords.append(frame)
        self.events_actions.append('rearing_start')
        self.open_rearing = True
        self.update_table()
        self._update_status_bar()
        self.statusLabel.setText(f"{self.current_behavior.capitalize()} start at frame {frame}")
        self.statusLabel.setStyleSheet("color: green; font-weight: bold; padding: 5px;")

    def add_rearing_end(self):
        """Mark rearing end event at current frame."""
        if not self.open_rearing:
            self.statusLabel.setText("No open event to close")
            self.statusLabel.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
            return

        self._save_undo_state()
        frame = self.get_current_frame()
        self.events_coords.append(frame)
        self.events_actions.append('rearing_end')
        self.events_labels.append(self.current_behavior)
        self.open_rearing = False
        self.update_table()
        self.write_csv()
        self._update_status_bar()
        self.statusLabel.setText(f"{self.current_behavior.capitalize()} end at frame {frame} - Saved!")
        self.statusLabel.setStyleSheet("color: blue; font-weight: bold; padding: 5px;")

    def on_behavior_changed(self, behavior_text):
        """Callback when behavior selector changes."""
        if not self.open_rearing:
            self.current_behavior = behavior_text
            self._update_status_bar()
            self.statusLabel.setText(f"Behavior changed to: {behavior_text.capitalize()}")
            self.statusLabel.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
        else:
            self.behaviorComboBox.setCurrentText(self.current_behavior)
            self.statusLabel.setText("Cannot change behavior while event is open. Close current event first.")
            self.statusLabel.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")

    def select_behavior_by_index(self, index: int):
        """Select behavior by index (0-based) using keyboard shortcut.

        Args:
            index: 0-based index into the behavior list
        """
        if self.open_rearing:
            self.statusLabel.setText("Cannot change behavior while event is open. Close current event first.")
            self.statusLabel.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
            return

        if index < self.behaviorComboBox.count():
            self.behaviorComboBox.setCurrentIndex(index)
            behavior = self.behaviorComboBox.currentText()
            self.current_behavior = behavior
            self._update_status_bar()
            self.statusLabel.setText(f"[{index+1}] Selected: {behavior.capitalize()}")
            self.statusLabel.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
        else:
            self.statusLabel.setText(f"No behavior at index {index+1}")
            self.statusLabel.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")

    def update_table(self):
        """Update the table with current rearing events."""
        # Group start/end pairs with labels
        events = []
        event_idx = 0
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
                    # Get label for this event (use stored label or default to current_behavior)
                    label = self.events_labels[event_idx] if event_idx < len(self.events_labels) else self.current_behavior
                    events.append((start_frame, end_frame, duration, label))
                    event_idx += 1
                else:
                    # Incomplete event
                    events.append((start_frame, None, None, self.current_behavior))
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
        
        # Clear and repopulate table (block signals during update)
        self._table_updating = True
        self.tableWidget.setRowCount(0)
        self.tableWidget.setRowCount(len(events))

        for row, event_data in enumerate(events):
            if len(event_data) == 4:
                start, end, duration, label = event_data
            else:
                start, end, duration = event_data[:3]
                label = self.current_behavior

            # Index (not editable)
            item_idx = QTableWidgetItem(str(row + 1))
            item_idx.setFlags(item_idx.flags() & ~Qt.ItemIsEditable)
            self.tableWidget.setItem(row, 0, item_idx)

            # Start frame (editable for complete events)
            item_start = QTableWidgetItem(str(start))
            if end is None:
                item_start.setFlags(item_start.flags() & ~Qt.ItemIsEditable)
            self.tableWidget.setItem(row, 1, item_start)

            if end is not None:
                # End frame (editable)
                item_end = QTableWidgetItem(str(end))
                self.tableWidget.setItem(row, 2, item_end)

                # Duration (not editable, computed)
                item_dur = QTableWidgetItem(f"{duration} frames")
                item_dur.setFlags(item_dur.flags() & ~Qt.ItemIsEditable)
                self.tableWidget.setItem(row, 3, item_dur)

                # Label (not editable for now)
                item_label = QTableWidgetItem(str(label))
                item_label.setFlags(item_label.flags() & ~Qt.ItemIsEditable)
                self.tableWidget.setItem(row, 4, item_label)
            else:
                # Incomplete event (not editable)
                item_end = QTableWidgetItem("...")
                item_end.setFlags(item_end.flags() & ~Qt.ItemIsEditable)
                self.tableWidget.setItem(row, 2, item_end)

                item_dur = QTableWidgetItem("open")
                item_dur.setFlags(item_dur.flags() & ~Qt.ItemIsEditable)
                self.tableWidget.setItem(row, 3, item_dur)

                item_label = QTableWidgetItem(str(label))
                item_label.setFlags(item_label.flags() & ~Qt.ItemIsEditable)
                self.tableWidget.setItem(row, 4, item_label)

                # Highlight incomplete events
                for col in range(5):
                    item = self.tableWidget.item(row, col)
                    if item:
                        item.setBackground(QBrush(QColor(255, 255, 0, 100)))

        self._table_updating = False
        
        # Update plots after table update
        self.update_plot()
        # Update feature plot window if open
        if self.feature_plot_window is not None and self.feature_plot_window.isVisible():
            self._last_feature_plot_update_frame = None  # Force full redraw when events change
            self.update_feature_plot()

    def _on_table_cell_edited(self, row: int, col: int):
        """Handle manual edits to table cells (start/end frame columns).

        Args:
            row: Table row index
            col: Table column index (1=start, 2=end)
        """
        if self._table_updating:
            return
        if col not in (1, 2):
            return

        item = self.tableWidget.item(row, col)
        if item is None:
            return

        try:
            new_value = int(item.text())
            if new_value < 0:
                raise ValueError("Frame must be non-negative")
        except ValueError:
            self.statusLabel.setText("Invalid frame number")
            self.statusLabel.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
            self.update_table()  # Restore original values
            return

        # Find the event index in events_coords (pairs of start/end)
        event_pairs = []
        i = 0
        while i < len(self.events_coords):
            if i < len(self.events_actions) and self.events_actions[i] == 'rearing_start':
                start_idx = i
                end_idx = None
                for j in range(i + 1, len(self.events_coords)):
                    if j < len(self.events_actions) and self.events_actions[j] == 'rearing_end':
                        end_idx = j
                        break
                if end_idx is not None:
                    event_pairs.append((start_idx, end_idx))
                i += 1
            else:
                i += 1

        if row >= len(event_pairs):
            return

        start_idx, end_idx = event_pairs[row]

        # Validate: start must be < end
        if col == 1:  # Editing start frame
            end_val = self.events_coords[end_idx]
            if new_value >= end_val:
                self.statusLabel.setText("Start must be before end")
                self.statusLabel.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
                self.update_table()
                return
            target_idx = start_idx
        else:  # Editing end frame
            start_val = self.events_coords[start_idx]
            if new_value <= start_val:
                self.statusLabel.setText("End must be after start")
                self.statusLabel.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
                self.update_table()
                return
            target_idx = end_idx

        # Save undo state and apply change
        self._save_undo_state()
        self.events_coords[target_idx] = new_value
        self.update_table()
        self.write_csv()
        self._update_status_bar()
        self.statusLabel.setText(f"Updated {'start' if col == 1 else 'end'} frame to {new_value}")
        self.statusLabel.setStyleSheet("color: green; font-weight: bold; padding: 5px;")

    def update_plot(self):
        """Update the matplotlib plot with current events (no features here)."""
        # Get complete events with labels for color coding
        events = self._get_complete_events(include_labels=True)
        # Also get events without labels for backwards compatibility in some places
        events_no_labels = [(e[0], e[1]) for e in events]

        # Legacy parsing removed - using _get_complete_events now
        i = 0
        if False:  # Keeping structure to minimize diff
            pass
        else:
                i += 1
        
        # Get current frame for indicator
        current_frame = self.get_current_frame()
        
        # Determine x-axis limits based on zoom
        if self.plot_zoom_factor == 1.0:
            # Full view: show all frames
            if self.frame_count > 0:
                x_min, x_max = 0, self.frame_count
            elif events:
                # Events can be (start, end) or (start, end, label) tuples
                max_frame = max(event[1] for event in events if event[1] is not None)
                x_min, x_max = 0, max(max_frame + 100, 100)
            elif self.features_df is not None and len(self.features_df) > 0:
                x_min, x_max = 0, len(self.features_df)
            else:
                x_min, x_max = 0, max(current_frame + 100, 100)
        else:
            # Zoomed view: show N frames around center
            half_window = self.zoom_n_frames // 2
            x_min = max(0, self.plot_center_frame - half_window)
            x_max = self.plot_center_frame + half_window
        
        # Initialize plot if needed (first time)
        if not hasattr(self, 'ax') or self.ax is None:
            self.fig.set_size_inches(8, 2)
            self.canvas.setMinimumHeight(150)
            self.ax = self.fig.add_subplot(111)
            self.axes = [self.ax]
            self.ax.set_xlabel('Frame')
            self.ax.set_ylabel('Events')
            self.ax.set_ylim(0, 1)
            self.ax.grid(True, alpha=0.3)
            self.event_rectangles = []
            self.frame_indicator_line = None
            self.zoom_indicator_line = None
        
        # Remove old event rectangles
        for rect in self.event_rectangles:
            rect.remove()
        self.event_rectangles.clear()

        # Draw new event bars with color coding by behavior
        for event in events:
            start, end = event[0], event[1]
            label = event[2] if len(event) > 2 else self.current_behavior
            facecolor, edgecolor = self._get_behavior_color(label)
            width = end - start
            rect = Rectangle((start, 0), width, 1, facecolor=facecolor, alpha=0.4,
                           edgecolor=edgecolor, linewidth=0.5)
            self.ax.add_patch(rect)
            self.event_rectangles.append(rect)
        
        # Update or create current frame indicator
        if self.frame_indicator_line is not None:
            self.frame_indicator_line.remove()
            self.frame_indicator_line = None
        
        if x_min <= current_frame <= x_max:
            self.frame_indicator_line = self.ax.axvline(x=current_frame, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        
        # Update or create zoom center indicator
        if self.zoom_indicator_line is not None:
            self.zoom_indicator_line.remove()
            self.zoom_indicator_line = None
        
        if self.plot_zoom_factor != 1.0 and x_min <= self.plot_center_frame <= x_max:
            self.zoom_indicator_line = self.ax.axvline(x=self.plot_center_frame, color='green', linestyle=':', linewidth=1, alpha=0.5)
        
        # Update x-axis limits and title
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_title(f'Rearing Events (Zoom: {self.zoom_n_frames} frames, Center: {self.plot_center_frame})')
        
        # Use draw_idle for smoother updates
        self.canvas.draw_idle()

    def on_plot_click(self, event):
        """Handle clicks on any plot (main or feature) to jump to video frame.

        Checks main plot axes, then feature plot axes if the click wasn't on main plot.
        """
        if event.xdata is None:
            return

        # Check if click is in main plot axes
        clicked_ax = None
        if hasattr(self, 'axes') and self.axes:
            for ax in self.axes:
                if event.inaxes == ax:
                    clicked_ax = ax
                    break
        if clicked_ax is None and hasattr(self, 'ax') and event.inaxes == self.ax:
            clicked_ax = self.ax

        # Check if click is in feature plot axes
        if clicked_ax is None and hasattr(self, 'feature_axes') and self.feature_axes:
            for ax in self.feature_axes:
                if event.inaxes == ax:
                    clicked_ax = ax
                    break

        if clicked_ax is None:
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
            # Reset session and features when opening new video
            self.current_session = None
            self.features_df = None
            self.selected_features = []
            self.update_feature_selection_ui()
            # Check for existing CSV and load it, or initialize new one
            self.check_and_load_csv()
            self.update_plot()

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
        self.events_labels = []
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
                header = next(reader, None)  # Get header to check format
                has_label = header and 'label' in [col.lower() for col in header]
                
                for i, row in enumerate(reader):
                    if len(row) >= 3:  # index, start_frame, end_frame, [duration, label]
                        try:
                            start_frame = int(row[1])
                            end_frame = int(row[2])
                            # Get label if present, otherwise use current_behavior
                            label = row[4].strip() if has_label and len(row) > 4 and row[4].strip() else self.current_behavior
                            
                            self.events_coords.append(start_frame)
                            self.events_actions.append('rearing_start')
                            self.events_coords.append(end_frame)
                            self.events_actions.append('rearing_end')
                            self.events_labels.append(label)
                        except ValueError:
                            continue
                self.update_table()

    def import_features(self):
        """Import features CSV or Excel file for current session."""
        if self.current_session is None:
            self.errorLabel.setText("No session found. Import features from project session.")
            self.errorLabel.setStyleSheet('color: red')
            return
        
        # Open file dialog (support both CSV and Excel)
        fileName, _ = QFileDialog.getOpenFileName(
            self, 
            "Import Features", 
            QDir.homePath(), 
            "All Supported (*.csv *.xlsx *.xls);;CSV Files(*.csv);;Excel Files(*.xlsx *.xls)"
        )
        
        if fileName:
            try:
                # Validate and import features
                features_path = Path(fileName)
                video_frame_count = self.frame_count if self.frame_count > 0 else None
                
                imported_path = self.current_session.import_features(features_path, video_frame_count)
                
                # Load the imported features
                self.load_features_from_path(imported_path)
                
                self.statusLabel.setText(f"Imported features: {features_path.name}. Right-click button to select features.")
                self.statusLabel.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
            except Exception as e:
                self.errorLabel.setText(f"Error importing features: {e}")
                self.errorLabel.setStyleSheet('color: red')
    
    def load_features_for_session(self):
        """Load features if they exist for current session."""
        if self.current_session is None:
            return
        
        features_path = self.current_session.features_path()
        if features_path.exists():
            self.load_features_from_path(features_path)
    
    def load_features_from_path(self, features_path: Path):
        """Load features from CSV or Excel file."""
        try:
            file_ext = features_path.suffix.lower()
            if file_ext in ['.xlsx', '.xls']:
                self.features_df = pd.read_excel(features_path)
            elif file_ext == '.csv':
                self.features_df = pd.read_csv(features_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            # Skip validation - allow features of any length
            
            # Update feature selection UI
            self.update_feature_selection_ui()
            
            # Update plots (catch any errors to prevent UI freezing)
            try:
                self.update_plot()
            except Exception as plot_error:
                print(f"Warning: Error updating plot: {plot_error}")
                # Don't fail feature loading if plot update fails
            
            # Show button to open feature plot window
            self.viewFeaturesButton.setVisible(True)
            
            # If feature plot window is open, update it (catch errors)
            if self.feature_plot_window is not None and self.feature_plot_window.isVisible():
                try:
                    self.update_feature_plot()
                except Exception as plot_error:
                    print(f"Warning: Error updating feature plot: {plot_error}")
            
            # Show message
            num_features = len(self.features_df.columns)
            self.statusLabel.setText(f"Loaded {num_features} features ({len(self.features_df)} frames)")
            self.statusLabel.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
            
        except Exception as e:
            import traceback
            error_msg = f"Error loading features: {e}"
            print(f"{error_msg}\n{traceback.format_exc()}")
            self.errorLabel.setText(error_msg)
            self.errorLabel.setStyleSheet('color: red')
            self.features_df = None
    
    def update_feature_selection_ui(self):
        """Update the feature selection checkboxes."""
        # Clear existing checkboxes
        while self.featureCheckboxLayout.count():
            child = self.featureCheckboxLayout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if self.features_df is None:
            self.feature_checkboxes = {}
            return
        
        # Create checkbox for each feature column
        self.feature_checkboxes = {}
        for col in self.features_df.columns:
            checkbox = QCheckBox(col)
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self.on_feature_selection_changed)
            self.featureCheckboxLayout.addWidget(checkbox)
            self.feature_checkboxes[col] = checkbox
    
    def on_feature_selection_changed(self):
        """Handle feature selection checkbox changes."""
        if hasattr(self, 'feature_checkboxes'):
            self.selected_features = [
                name for name, checkbox in self.feature_checkboxes.items()
                if checkbox.isChecked()
            ]
            # Update feature plot window if open
            if self.feature_plot_window is not None and self.feature_plot_window.isVisible():
                self._last_feature_plot_update_frame = None  # Force full redraw
                self.update_feature_plot()
    
    def show_feature_menu(self, position):
        """Show context menu for feature selection."""
        if self.features_df is None:
            return
        
        menu = QMenu(self)
        
        # Add "Select Features..." action
        select_action = QAction("Select Features...", self)
        select_action.triggered.connect(self.open_feature_selection_dialog)
        menu.addAction(select_action)
        
        # Add separator
        menu.addSeparator()
        
        # Add each feature as a checkable action
        for col in self.features_df.columns:
            action = QAction(col, self)
            action.setCheckable(True)
            action.setChecked(col in self.selected_features)
            action.triggered.connect(lambda checked, c=col: self.toggle_feature(c, checked))
            menu.addAction(action)
        
        # Show menu at button position
        button_pos = self.importFeaturesButton.mapToGlobal(position)
        menu.exec_(button_pos)
    
    def toggle_feature(self, feature_name: str, checked: bool):
        """Toggle a feature in the selected list."""
        if checked and feature_name not in self.selected_features:
            self.selected_features.append(feature_name)
        elif not checked and feature_name in self.selected_features:
            self.selected_features.remove(feature_name)
        
        # Update feature plot window if open
        if self.feature_plot_window is not None and self.feature_plot_window.isVisible():
            self._last_feature_plot_update_frame = None  # Force full redraw
            self.update_feature_plot()
    
    def open_feature_selection_dialog(self):
        """Open a dialog to select features."""
        if self.features_df is None:
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Features to Plot")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout()
        
        # Create scroll area with checkboxes
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        for col in self.features_df.columns:
            checkbox = QCheckBox(col)
            checkbox.setChecked(col in self.selected_features)
            checkbox.stateChanged.connect(lambda state, c=col: self.toggle_feature(c, state == Qt.Checked))
            scroll_layout.addWidget(checkbox)
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: self.select_all_features(True, scroll_widget))
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(lambda: self.select_all_features(False, scroll_widget))
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
        # Update feature plot window if open
        if self.feature_plot_window is not None and self.feature_plot_window.isVisible():
            self._last_feature_plot_update_frame = None  # Force full redraw
            self.update_feature_plot()
    
    def select_all_features(self, select: bool, widget: QWidget):
        """Select or deselect all features in the dialog."""
        for child in widget.findChildren(QCheckBox):
            child.setChecked(select)
    
    def open_feature_plot_window(self):
        """Open separate window for feature plots.

        Creates a new window with matplotlib canvas for visualizing selected features
        alongside annotated events. Clicking on the plot jumps video to that frame.
        """
        if self.features_df is None or len(self.selected_features) == 0:
            self.errorLabel.setText("No features selected. Select features from checkboxes first.")
            self.errorLabel.setStyleSheet('color: orange')
            return

        # Create feature plot window if it doesn't exist
        if self.feature_plot_window is None:
            self.feature_plot_window = QWidget()
            self.feature_plot_window.setWindowTitle("Feature Plot Viewer")
            self.feature_plot_window.setMinimumSize(800, 600)

            main_layout = QVBoxLayout()
            main_layout.setContentsMargins(0, 0, 0, 0)

            self.feature_fig = Figure(figsize=(10, 6), facecolor='white')
            self.feature_canvas = FigureCanvas(self.feature_fig)
            self.feature_canvas.setStyleSheet("background-color: white;")
            self.feature_ax = None

            # Connect click handler for click-to-jump functionality
            self.feature_canvas.mpl_connect('button_press_event', self.on_plot_click)

            main_layout.addWidget(self.feature_canvas)
            self.feature_plot_window.setLayout(main_layout)

        # Reset state for clean redraw
        self._reset_feature_plot_state()

        # Clear figure before showing to prevent artifacts
        self.feature_fig.clf()

        self.feature_plot_window.show()
        self.update_feature_plot()
    
    def _reset_feature_plot_state(self):
        """Reset feature plot state for clean redraw.

        Clears all tracked plot objects and forces next update to do full redraw.
        """
        self._last_feature_plot_update_frame = None
        self._last_drawn_events_hash = None

        # Safely clear tracked plot objects
        if hasattr(self, 'feature_lines'):
            for line in self.feature_lines:
                if line is not None:
                    try:
                        line.remove()
                    except ValueError:
                        pass  # Already removed
            self.feature_lines = []

        if hasattr(self, 'feature_event_rectangles'):
            for rect_list in self.feature_event_rectangles:
                for rect in rect_list:
                    if rect is not None:
                        try:
                            rect.remove()
                        except ValueError:
                            pass
            self.feature_event_rectangles = []

        if hasattr(self, 'feature_frame_indicators'):
            for line in self.feature_frame_indicators:
                if line is not None:
                    try:
                        line.remove()
                    except ValueError:
                        pass
            self.feature_frame_indicators = []

        self.feature_axes = []

    def update_feature_plot(self):
        """Update the feature plot window with current data and frame position.

        Uses incremental updates when possible (frame indicator only) for performance,
        falls back to full redraw when structure changes (features added/removed).
        """
        if self.feature_plot_window is None or self.feature_canvas is None:
            return
        if self.features_df is None or len(self.selected_features) == 0:
            return

        current_frame = self.get_current_frame()
        num_features = len(self.selected_features)
        events = self._get_complete_events(include_labels=True)
        events_hash = hash(tuple((e[0], e[1]) for e in events)) if events else 0

        # Initialize state if needed
        if not hasattr(self, 'feature_lines'):
            self.feature_lines = []
            self.feature_event_rectangles = []
            self.feature_frame_indicators = []
            self.feature_axes = []
            self._last_feature_plot_update_frame = None
            self._last_drawn_events_hash = None

        # Check if we can do incremental update (structure unchanged)
        structure_valid = (
            len(self.feature_axes) == num_features and
            len(self.feature_lines) == num_features and
            all(line is not None for line in self.feature_lines)
        )

        if structure_valid and self._last_feature_plot_update_frame is not None:
            self._update_feature_plot_incremental(current_frame, events, events_hash)
        else:
            self._update_feature_plot_full(current_frame, events, events_hash, num_features)

    def _update_feature_plot_incremental(self, current_frame: int, events: list, events_hash: int):
        """Incremental update: move frame indicators and update events if changed.

        Only updates frame indicator positions without redrawing data lines.
        Falls back to full redraw if state is inconsistent.

        Args:
            current_frame: Current video frame number
            events: List of (start, end, label) tuples
            events_hash: Hash of events for change detection
        """
        # Validate state before incremental update
        if len(self.feature_frame_indicators) != len(self.feature_axes):
            # State mismatch - do full redraw instead
            self._update_feature_plot_full(current_frame, events, events_hash, len(self.selected_features))
            return

        events_changed = events_hash != self._last_drawn_events_hash

        # Update frame indicators (fast path using set_xdata)
        for idx, ax in enumerate(self.feature_axes):
            indicator = self.feature_frame_indicators[idx]
            if indicator is not None:
                y_min, y_max = ax.get_ylim()
                indicator.set_xdata([current_frame, current_frame])
                indicator.set_ydata([y_min, y_max])

        # Update event rectangles only if events changed
        if events_changed:
            self._redraw_event_rectangles(events)
            self._last_drawn_events_hash = events_hash

        self.feature_canvas.draw_idle()
        self._last_feature_plot_update_frame = current_frame

    def _update_feature_plot_full(self, current_frame: int, events: list, events_hash: int, num_features: int):
        """Full redraw of feature plot (structure rebuild).

        Args:
            current_frame: Current video frame number
            events: List of (start, end, label) tuples
            events_hash: Hash of events for change detection
            num_features: Number of features to plot
        """
        # Clear all tracked objects first
        self.feature_lines = []
        self.feature_event_rectangles = []
        self.feature_frame_indicators = []
        self.feature_axes = []

        # Thorough figure clear: remove all axes and reset
        for ax in self.feature_fig.axes[:]:
            self.feature_fig.delaxes(ax)
        self.feature_fig.clf()

        # Set figure size before creating subplots to avoid resize artifacts
        fig_height = min(max(3, 2 * num_features), 12)
        self.feature_fig.set_size_inches(8, fig_height)

        # Create subplots using subplots() for cleaner layout
        if num_features == 1:
            axes_list = [self.feature_fig.add_subplot(1, 1, 1)]
        else:
            axes_list = []
            for idx in range(num_features):
                ax = self.feature_fig.add_subplot(num_features, 1, idx + 1)
                axes_list.append(ax)

        for idx, feature_name in enumerate(self.selected_features):
            ax = axes_list[idx]
            self.feature_axes.append(ax)

            # Plot feature data
            feature_data = self.features_df[feature_name].values
            frames = np.arange(len(feature_data))
            line, = ax.plot(frames, feature_data, 'b-', linewidth=1.0, alpha=0.8)
            self.feature_lines.append(line)

            # Set y-axis limits from data
            valid_data = feature_data[~np.isnan(feature_data)]
            if len(valid_data) > 0:
                data_range = valid_data.max() - valid_data.min()
                margin = data_range * 0.1 if data_range > 0 else 0.1
                ax.set_ylim(valid_data.min() - margin, valid_data.max() + margin)
            else:
                ax.set_ylim(-1, 1)

            # Set x-axis limits to data range
            ax.set_xlim(0, len(feature_data))

            # Draw event rectangles
            rects = []
            y_min, y_max = ax.get_ylim()
            for event in events:
                start, end = event[0], event[1]
                label = event[2] if len(event) > 2 else self.current_behavior
                facecolor, edgecolor = self._get_behavior_color(label)
                rect = Rectangle((start, y_min), end - start, y_max - y_min,
                                  facecolor=facecolor, alpha=0.3, edgecolor=edgecolor, linewidth=0.5)
                ax.add_patch(rect)
                rects.append(rect)
            self.feature_event_rectangles.append(rects)

            # Draw frame indicator
            if 0 <= current_frame < len(feature_data):
                indicator = ax.axvline(x=current_frame, color='blue', linestyle='--', linewidth=2, alpha=0.7)
                self.feature_frame_indicators.append(indicator)
            else:
                self.feature_frame_indicators.append(None)

            # Labels and formatting
            ax.set_ylabel(feature_name, fontsize=9)
            ax.grid(True, alpha=0.3)
            if idx == num_features - 1:
                ax.set_xlabel('Frame')
            else:
                ax.tick_params(labelbottom=False)

        # Title
        if num_features == 1:
            self.feature_fig.suptitle(f'{self.selected_features[0]} with Events', fontsize=10)
        else:
            self.feature_fig.suptitle('Selected Features with Events', fontsize=10)

        # Layout and draw
        self.feature_fig.tight_layout()
        self.feature_canvas.draw()
        self.feature_canvas.flush_events()

        self._last_drawn_events_hash = events_hash
        self._last_feature_plot_update_frame = current_frame

    def _redraw_event_rectangles(self, events: list):
        """Redraw event rectangles on all feature axes.

        Args:
            events: List of (start, end, label) tuples
        """
        # Remove old rectangles
        for rect_list in self.feature_event_rectangles:
            for rect in rect_list:
                if rect is not None:
                    try:
                        rect.remove()
                    except ValueError:
                        pass
        self.feature_event_rectangles = []

        # Draw new rectangles
        for idx, ax in enumerate(self.feature_axes):
            rects = []
            y_min, y_max = ax.get_ylim()
            for event in events:
                start, end = event[0], event[1]
                label = event[2] if len(event) > 2 else self.current_behavior
                facecolor, edgecolor = self._get_behavior_color(label)
                rect = Rectangle((start, y_min), end - start, y_max - y_min,
                                  facecolor=facecolor, alpha=0.3, edgecolor=edgecolor, linewidth=0.5)
                ax.add_patch(rect)
                rects.append(rect)
            self.feature_event_rectangles.append(rects)
    
    def _get_complete_events(self, include_labels=False):
        """Helper to get complete rearing events as (start, end) pairs or (start, end, label) tuples.

        Args:
            include_labels: If True, return (start, end, label) tuples
        """
        events = []
        event_idx = 0
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
                    if include_labels:
                        label = self.events_labels[event_idx] if event_idx < len(self.events_labels) else self.current_behavior
                        events.append((start_frame, end_frame, label))
                    else:
                        events.append((start_frame, end_frame))
                    event_idx += 1
                i += 1
            else:
                i += 1
        return events

    def _get_behavior_color(self, behavior: str):
        """Get the color tuple (facecolor, edgecolor) for a behavior.

        Args:
            behavior: Behavior name string

        Returns:
            Tuple of (facecolor, edgecolor) hex strings
        """
        behavior_lower = behavior.lower()
        if behavior_lower in self.behavior_colors:
            return self.behavior_colors[behavior_lower]
        return self.default_color

    def insertBaseRow(self):
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setHorizontalHeaderLabels(["Index", "Start Frame", "End Frame", "Duration", "Label"])
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
        # Update plot when position changes (for current frame indicator)
        # This handles automatic position changes (playback, etc.)
        if hasattr(self, 'canvas'):
            current_frame = self.get_current_frame()
            # Throttle updates: every 5 frames when zoomed, every 30 frames when not zoomed
            update_threshold = 5 if self.plot_zoom_factor != 1.0 else 30
            if hasattr(self, '_last_plot_update_frame'):
                if abs(current_frame - self._last_plot_update_frame) > update_threshold:
                    self.update_plot()
                    self._last_plot_update_frame = current_frame
                    # Update feature plot window if open (throttled)
                    if self.feature_plot_window is not None and self.feature_plot_window.isVisible():
                        self.update_feature_plot()
            else:
                self._last_plot_update_frame = current_frame
                self.update_plot()
                # Update feature plot window if open
                if self.feature_plot_window is not None and self.feature_plot_window.isVisible():
                    self.update_feature_plot()

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
        # Update plot when slider moves (for current frame indicator)
        if hasattr(self, 'canvas'):
            current_frame = self.get_current_frame()
            # Update plot to show current frame indicator
            # Throttle updates: every 5 frames when zoomed, every 30 frames when not zoomed
            update_threshold = 5 if self.plot_zoom_factor != 1.0 else 30
            if hasattr(self, '_last_plot_update_frame'):
                if abs(current_frame - self._last_plot_update_frame) > update_threshold:
                    self.update_plot()
                    self._last_plot_update_frame = current_frame
                    # Update feature plot window if open (throttled)
                    if self.feature_plot_window is not None and self.feature_plot_window.isVisible():
                        self.update_feature_plot()
            else:
                self._last_plot_update_frame = current_frame
                self.update_plot()
                # Update feature plot window if open
                if self.feature_plot_window is not None and self.feature_plot_window.isVisible():
                    self.update_feature_plot()

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
                    # Update feature plot window if open
                    if self.feature_plot_window is not None and self.feature_plot_window.isVisible():
                        self.update_feature_plot()
            else:
                self._last_plot_update_frame = frame
                self.update_plot()
                # Update feature plot window if open
                if self.feature_plot_window is not None and self.feature_plot_window.isVisible():
                    self.update_feature_plot()

    def dropEvent(self, event):
        f = str(event.mimeData().urls()[0].toLocalFile())
        self.load_video_from_path(Path(f))
    
    def closeEvent(self, event):
        """Handle window close event - save annotations before closing."""
        # Save any open events and write CSV
        if len(self.events_coords) > 0:
            self.write_csv()
            print("Annotations saved before closing.")
        # Don't call sys.exit here - just close the window
        event.accept()
    
    def switch_session(self):
        """Switch to a different session for annotation."""
        if self.project is None or self.sessions is None:
            return
        
        # Save current annotations before switching
        if len(self.events_coords) > 0:
            self.write_csv()
            print("Saved annotations before switching session.")
        
        # Show session selection dialog
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton
        from PyQt5.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Switch to Another Session")
        dialog.setMinimumSize(600, 400)
        
        layout = QVBoxLayout()
        dialog.setLayout(layout)
        
        label = QLabel(f"Select a session to annotate:")
        layout.addWidget(label)
        
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SingleSelection)
        
        # Add sessions to list
        for session in self.sessions:
            item_text = f"{session.id} - {session.annotation_view} view"
            if hasattr(session, 'video_dir') and session.video_dir:
                video_dir = str(session.video_dir)
                if len(video_dir) > 50:
                    video_dir = "..." + video_dir[-47:]
                item_text += f"\n  {video_dir}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, session)
            list_widget.addItem(item)
        
        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)
        
        layout.addWidget(list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        
        switch_button = QPushButton("Switch to Selected Session")
        switch_button.setDefault(True)
        switch_button.clicked.connect(dialog.accept)
        button_layout.addWidget(switch_button)
        
        layout.addLayout(button_layout)
        
        # Show dialog and switch if accepted
        if dialog.exec_() == QDialog.Accepted:
            selected_items = list_widget.selectedItems()
            if selected_items:
                selected_session = selected_items[0].data(Qt.UserRole)
                try:
                    annotation_view_path = selected_session.path_to_view(selected_session.annotation_view)
                    # Set current session before loading
                    self.current_session = selected_session
                    # Load new video (will also load features if available)
                    self.load_video_from_path(annotation_view_path)
                    print(f"Switched to session {selected_session.id} ({selected_session.annotation_view} view)")
                except Exception as e:
                    self.errorLabel.setText(f"Error switching session: {e}")
                    self.errorLabel.setStyleSheet('color: red')


def start_app(video_path: Path | None = None, project=None, sessions=None, run_event_loop=None):
    """
    Create and run the rearing event annotator.
    
    Args:
        video_path: Optional path to video file to auto-load
        project: Optional Project instance for session switching
        sessions: Optional list of sessions for session switching
        run_event_loop: If False, don't start event loop (use when called from another GUI).
                       If None (default), auto-detect based on context.
    
    Usage:
        from behavex.annotation.pavs import start_app
        start_app(Path("/path/to/video.mp4"))
    """
    app = QApplication.instance()
    app_existed = app is not None
    if app is None:
        app = QApplication(sys.argv)
    
    window = Window(video_path=video_path, project=project, sessions=sessions)
    window.show()
    
    # Auto-detect if we should run event loop (only if not explicitly set)
    if run_event_loop is None:
        # Check if we're the main script (pavs.py run directly)
        is_main_script = (hasattr(sys.modules.get('__main__', None), '__file__') and 
                         sys.modules['__main__'].__file__ and
                         'pavs.py' in sys.modules['__main__'].__file__)
        
        # If app already existed, we're likely being called from another GUI
        # Check if there are other visible top-level windows (besides ours)
        if app_existed:
            try:
                from PyQt5.QtWidgets import QWidget
                # Get all top-level widgets
                # Since we just showed window, count visible top-level widgets
                all_widgets = app.allWidgets()
                visible_top_level = [w for w in all_widgets 
                                    if isinstance(w, QWidget) and w.isVisible() and w.parent() is None]
                # If there are multiple visible windows, event loop is likely running
                if len(visible_top_level) > 1:
                    run_event_loop = False
                else:
                    # Only one window (ours), safe to run
                    run_event_loop = True
            except Exception:
                # On error, be conservative - don't run if app existed
                run_event_loop = False
        else:
            # New app created, safe to run event loop
            run_event_loop = True
    
    # Only call exec_() if explicitly True
    # If False (explicitly set) or auto-detected as False, never call exec_()
    if run_event_loop is True:
        app.exec_()


if __name__ == "__main__":
    video_path_arg = None
    if len(sys.argv) > 1:
        video_path_arg = Path(sys.argv[1])
    
    start_app(video_path_arg)
