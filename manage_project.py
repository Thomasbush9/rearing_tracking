#!/usr/bin/env python3
"""
Interactive script for managing BehaveX projects: adding sessions and annotating behaviors.
"""
import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QListWidget, QListWidgetItem, QFileDialog,
    QLineEdit, QTextEdit, QMessageBox, QInputDialog
)
from PyQt5.QtCore import Qt
from behavex.project.project import Project


class ProjectManagerDialog(QDialog):
    """Main dialog for managing project sessions and annotations."""
    
    def __init__(self, project: Project = None, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("BehaveX Project Manager")
        self.setMinimumSize(700, 500)
        self.init_ui()
        self.update_display()
    
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Project info section
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        layout.addWidget(self.info_label)
        
        # Sessions list
        sessions_label = QLabel("Sessions in project:")
        layout.addWidget(sessions_label)
        
        self.sessions_list = QListWidget()
        self.sessions_list.setSelectionMode(QListWidget.SingleSelection)
        layout.addWidget(self.sessions_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.add_session_btn = QPushButton("Add Session")
        self.add_session_btn.clicked.connect(self.add_session)
        button_layout.addWidget(self.add_session_btn)
        
        self.config_behaviors_btn = QPushButton("Configure Behaviors")
        self.config_behaviors_btn.clicked.connect(self.configure_behaviors)
        button_layout.addWidget(self.config_behaviors_btn)
        
        self.annotate_btn = QPushButton("Annotate Sessions")
        self.annotate_btn.clicked.connect(self.annotate_sessions)
        self.annotate_btn.setEnabled(False)
        button_layout.addWidget(self.annotate_btn)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.update_display)
        button_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(button_layout)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
    
    def update_display(self):
        """Update the display with current project info."""
        if self.project is None:
            self.info_label.setText("No project loaded")
            self.sessions_list.clear()
            self.annotate_btn.setEnabled(False)
            return
        
        # Update info label
        session_count = len(self.project.sessions)
        self.info_label.setText(
            f"Project: {self.project.project_name}\n"
            f"Sessions: {session_count}"
        )
        
        # Update sessions list
        self.sessions_list.clear()
        for session in self.project.sessions:
            item_text = f"{session.id}"
            if session.annotation_view:
                item_text += f" ({session.annotation_view} view)"
            if hasattr(session, 'video_dir') and session.video_dir:
                video_dir = str(session.video_dir)
                if len(video_dir) > 60:
                    video_dir = "..." + video_dir[-57:]
                item_text += f"\n  {video_dir}"
            
            item = QListWidgetItem(item_text)
            self.sessions_list.addItem(item)
        
        # Enable annotate button if sessions exist
        self.annotate_btn.setEnabled(session_count > 0)
    
    def add_session(self):
        """Open dialog to add a new session."""
        if self.project is None:
            QMessageBox.warning(self, "No Project", "Please load or create a project first.")
            return
        
        # Select video directory
        video_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Video Directory",
            str(Path.home()),
            QFileDialog.ShowDirsOnly
        )
        
        if not video_dir:
            return
        
        video_dir_path = Path(video_dir)
        
        # Check for video files
        video_files = list(video_dir_path.glob("*.mp4"))
        if not video_files:
            QMessageBox.warning(
                self,
                "No Video Files",
                f"No .mp4 video files found in {video_dir_path}"
            )
            return
        
        try:
            # Add session
            session = self.project.add_session(video_dir_path)
            QMessageBox.information(
                self,
                "Session Added",
                f"Successfully added session: {session.id}\n"
                f"Views detected: {list(session.views.keys())}"
            )
            self.update_display()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to add session:\n{str(e)}"
            )
    
    def configure_behaviors(self):
        """Open dialog to configure behaviors."""
        if self.project is None:
            QMessageBox.warning(self, "No Project", "Please load or create a project first.")
            return
        
        dialog = BehaviorConfigDialog(self.project, self)
        dialog.exec_()
        self.update_display()
    
    def annotate_sessions(self):
        """Launch annotation GUI."""
        if self.project is None or len(self.project.sessions) == 0:
            QMessageBox.warning(
                self,
                "No Sessions",
                "No sessions available for annotation. Add sessions first."
            )
            return

        # Show session selection dialog first
        selected_session = self.select_session_for_annotation()
        if selected_session is None:
            return

        # Launch annotation window synchronously before closing dialog.
        # This ensures the window exists when main() checks for visible windows
        # after dialog.exec_() returns, preventing premature app exit.
        from behavex.annotation.pavs import start_app

        try:
            annotation_view_path = selected_session.path_to_view(selected_session.annotation_view)
            # run_event_loop=False: main() handles the event loop
            start_app(annotation_view_path, project=self.project, sessions=self.project.sessions, run_event_loop=False)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to start annotation:\n{str(e)}"
            )
            return

        # Close manager dialog after annotation window is created and visible
        self.accept()
    
    def select_session_for_annotation(self):
        """Show dialog to select a session for annotation."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem
        from PyQt5.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Session to Annotate")
        dialog.setMinimumSize(600, 400)
        
        layout = QVBoxLayout()
        dialog.setLayout(layout)
        
        label = QLabel(f"Select a session from {self.project.project_name}:")
        layout.addWidget(label)
        
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SingleSelection)
        
        for session in self.project.sessions:
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
        
        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        
        annotate_button = QPushButton("Annotate Selected Session")
        annotate_button.setDefault(True)
        annotate_button.clicked.connect(dialog.accept)
        button_layout.addWidget(annotate_button)
        
        layout.addLayout(button_layout)
        
        if dialog.exec_() == QDialog.Accepted:
            selected_items = list_widget.selectedItems()
            if selected_items:
                return selected_items[0].data(Qt.UserRole)
        
        return None


class BehaviorConfigDialog(QDialog):
    """Dialog for configuring behaviors."""
    
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Configure Behaviors")
        self.setMinimumSize(500, 400)
        self.init_ui()
        self.load_behaviors()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Instructions
        info_label = QLabel(
            "Configure behavior labels for annotation.\n"
            "Each behavior will appear in the annotation dropdown."
        )
        layout.addWidget(info_label)
        
        # Behaviors list
        list_label = QLabel("Behaviors:")
        layout.addWidget(list_label)
        
        self.behaviors_list = QListWidget()
        layout.addWidget(self.behaviors_list)
        
        # Buttons for managing behaviors
        button_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add Behavior")
        add_btn.clicked.connect(self.add_behavior)
        button_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_behavior)
        button_layout.addWidget(remove_btn)
        
        layout.addLayout(button_layout)
        
        # Default behaviors button
        default_btn = QPushButton("Use Default Behaviors")
        default_btn.clicked.connect(self.use_defaults)
        layout.addWidget(default_btn)
        
        # Save/Cancel buttons
        save_cancel_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_behaviors)
        save_cancel_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        save_cancel_layout.addWidget(cancel_btn)
        
        layout.addLayout(save_cancel_layout)
    
    def load_behaviors(self):
        """Load current behaviors from project config."""
        self.behaviors_list.clear()
        annotation_cfg = self.project.config.get("annotation", {})
        behaviors = annotation_cfg.get("behaviors", [])
        
        if not behaviors:
            # Try single behavior field
            single_behavior = annotation_cfg.get("behavior", None)
            if single_behavior:
                behaviors = [single_behavior]
            else:
                # Use defaults
                behaviors = ["rearing", "grooming", "exploring", "resting", "other"]
        
        for behavior in behaviors:
            self.behaviors_list.addItem(behavior)
    
    def add_behavior(self):
        """Add a new behavior."""
        text, ok = QInputDialog.getText(
            self,
            "Add Behavior",
            "Enter behavior name:",
            QLineEdit.Normal,
            ""
        )
        
        if ok and text:
            text = text.strip().lower()
            if text:
                # Check if already exists
                existing = [
                    self.behaviors_list.item(i).text() 
                    for i in range(self.behaviors_list.count())
                ]
                if text not in existing:
                    self.behaviors_list.addItem(text)
                else:
                    QMessageBox.information(
                        self,
                        "Duplicate",
                        f"Behavior '{text}' already exists."
                    )
    
    def remove_behavior(self):
        """Remove selected behavior."""
        current_item = self.behaviors_list.currentItem()
        if current_item:
            reply = QMessageBox.question(
                self,
                "Confirm Removal",
                f"Remove behavior '{current_item.text()}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.behaviors_list.takeItem(self.behaviors_list.row(current_item))
    
    def use_defaults(self):
        """Reset to default behaviors."""
        reply = QMessageBox.question(
            self,
            "Use Defaults",
            "Replace current behaviors with defaults?\n"
            "(rearing, grooming, exploring, resting, other)",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.behaviors_list.clear()
            for behavior in ["rearing", "grooming", "exploring", "resting", "other"]:
                self.behaviors_list.addItem(behavior)
    
    def save_behaviors(self):
        """Save behaviors to project config."""
        behaviors = [
            self.behaviors_list.item(i).text()
            for i in range(self.behaviors_list.count())
        ]
        
        if not behaviors:
            QMessageBox.warning(
                self,
                "No Behaviors",
                "Please add at least one behavior."
            )
            return
        
        self.project.set_config_value("annotation.behaviors", behaviors)
        QMessageBox.information(
            self,
            "Saved",
            f"Saved {len(behaviors)} behaviors:\n" + ", ".join(behaviors)
        )
        self.accept()


def load_or_create_project():
    """Dialog to load existing project or create new one."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Ask user what to do
    reply = QMessageBox.question(
        None,
        "Project Manager",
        "Load existing project or create new?",
        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
        QMessageBox.Yes
    )
    
    if reply == QMessageBox.Cancel:
        return None
    
    if reply == QMessageBox.Yes:
        # Load existing project
        project_dir = QFileDialog.getExistingDirectory(
            None,
            "Select Project Directory",
            str(Path.home()),
            QFileDialog.ShowDirsOnly
        )
        
        if not project_dir:
            return None
        
        try:
            project = Project(project_dir)
            return project
        except Exception as e:
            QMessageBox.critical(
                None,
                "Error",
                f"Failed to load project:\n{str(e)}"
            )
            return None
    else:
        # Create new project
        project_dir = QFileDialog.getExistingDirectory(
            None,
            "Select Directory for New Project",
            str(Path.home()),
            QFileDialog.ShowDirsOnly
        )
        
        if not project_dir:
            return None
        
        # Ask for project name
        project_name, ok = QInputDialog.getText(
            None,
            "New Project",
            "Enter project name:",
            QLineEdit.Normal,
            Path(project_dir).name
        )
        
        if not ok:
            return None
        
        try:
            project = Project(project_dir, project_name=project_name or None)
            QMessageBox.information(
                None,
                "Project Created",
                f"Created new project: {project.project_name}\n"
                f"Location: {project_dir}"
            )
            return project
        except Exception as e:
            QMessageBox.critical(
                None,
                "Error",
                f"Failed to create project:\n{str(e)}"
            )
            return None


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Load or create project
    project = load_or_create_project()
    
    if project is None:
        sys.exit(0)
    
    # Show main manager dialog
    dialog = ProjectManagerDialog(project)
    dialog.exec_()
    
    # After dialog closes, check if annotation window is still open
    # If so, keep event loop running
    from PyQt5.QtWidgets import QWidget
    visible_windows = [w for w in app.allWidgets() 
                      if isinstance(w, QWidget) and w.isVisible() and w.parent() is None]
    
    # If annotation window is open, keep event loop running
    if len(visible_windows) > 0:
        app.exec_()
    
    sys.exit(0)


if __name__ == "__main__":
    main()

