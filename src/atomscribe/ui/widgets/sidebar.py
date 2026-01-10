"""Sidebar widget with file browser - Notion style"""

from datetime import datetime
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QTreeView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLabel,
    QFrame,
    QFileSystemModel,
    QHeaderView,
)
from PySide6.QtCore import Qt, Signal, QDir
from PySide6.QtGui import QIcon, QFont

from ...signals import get_app_signals


class SidebarWidget(QWidget):
    """
    Left sidebar containing file browser.
    Notion-style design for browsing recording files.
    """

    file_selected = Signal(str)  # Emits file path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = get_app_signals()
        self.setObjectName("sidebar")
        self.setMinimumWidth(240)
        self.setMaximumWidth(300)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the sidebar UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sidebar header section
        header = QWidget()
        header.setObjectName("sidebarHeader")
        header.setStyleSheet("""
            QWidget#sidebarHeader {
                background-color: #FFFFFF;
                border-bottom: 1px solid #E8E8E8;
            }
        """)
        header.setFixedHeight(48)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 0, 16, 0)

        # Sidebar title
        title = QLabel("Files")
        title.setObjectName("sidebarTitle")
        title.setStyleSheet("""
            font-size: 13px;
            font-weight: 600;
            color: #37352F;
        """)
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Refresh button
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedSize(24, 24)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                color: #787774;
            }
            QPushButton:hover {
                background-color: #F0F0F0;
                color: #37352F;
            }
        """)
        refresh_btn.clicked.connect(self._refresh_files)
        refresh_btn.setToolTip("Refresh")
        header_layout.addWidget(refresh_btn)

        layout.addWidget(header)

        # Search box
        search_container = QWidget()
        search_container.setStyleSheet("background-color: #FFFFFF;")
        search_layout = QVBoxLayout(search_container)
        search_layout.setContentsMargins(12, 10, 12, 10)

        self.search_box = QLineEdit()
        self.search_box.setObjectName("searchBox")
        self.search_box.setPlaceholderText("Search files...")
        self.search_box.setStyleSheet("""
            QLineEdit {
                background-color: #F7F7F5;
                border: 1px solid #E8E8E8;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 12px;
                color: #37352F;
            }
            QLineEdit:focus {
                border-color: #2383E2;
                background-color: #FFFFFF;
            }
        """)
        self.search_box.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self.search_box)

        layout.addWidget(search_container)

        # File tree view
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath("")
        # Only show specific file types
        self.file_model.setNameFilters(["*.json", "*.txt", "*.md", "*.wav", "*.mp3"])
        self.file_model.setNameFilterDisables(False)

        self.file_tree = QTreeView()
        self.file_tree.setObjectName("fileTree")
        self.file_tree.setModel(self.file_model)
        self.file_tree.setHeaderHidden(True)
        # Hide all columns except name
        self.file_tree.setColumnHidden(1, True)  # Size
        self.file_tree.setColumnHidden(2, True)  # Type
        self.file_tree.setColumnHidden(3, True)  # Date
        self.file_tree.setAnimated(True)
        self.file_tree.setIndentation(16)
        self.file_tree.setStyleSheet("""
            QTreeView {
                background-color: #FFFFFF;
                border: none;
                outline: none;
                font-size: 12px;
            }
            QTreeView::item {
                padding: 4px 8px;
                border-radius: 4px;
                margin: 1px 4px;
            }
            QTreeView::item:hover {
                background-color: #F7F7F5;
            }
            QTreeView::item:selected {
                background-color: #E3F2FD;
                color: #1976D2;
            }
            QTreeView::branch {
                background-color: transparent;
            }
        """)
        self.file_tree.clicked.connect(self._on_file_clicked)

        # Set default path to current working directory or user home
        default_path = Path.cwd()
        if default_path.exists():
            self.file_tree.setRootIndex(self.file_model.index(str(default_path)))

        layout.addWidget(self.file_tree, stretch=1)

        # Bottom action section
        bottom_section = QWidget()
        bottom_section.setStyleSheet("""
            background-color: #FAFAFA;
            border-top: 1px solid #E8E8E8;
        """)
        bottom_layout = QVBoxLayout(bottom_section)
        bottom_layout.setContentsMargins(12, 10, 12, 10)

        # Open folder button
        self.open_folder_btn = QPushButton("Open Folder...")
        self.open_folder_btn.setObjectName("openFolderButton")
        self.open_folder_btn.clicked.connect(self._on_open_folder_clicked)
        self.open_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 8px 12px;
                color: #37352F;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #F7F7F5;
                border-color: #BDBDBD;
            }
        """)
        bottom_layout.addWidget(self.open_folder_btn)

        layout.addWidget(bottom_section)

    def _connect_signals(self):
        """Connect to app signals"""
        self.signals.file_selected.connect(self._on_external_file_selected)

    def _on_search_changed(self, text: str):
        """Filter files based on search text"""
        if text:
            self.file_model.setNameFilters([f"*{text}*"])
        else:
            self.file_model.setNameFilters(["*.json", "*.txt", "*.md", "*.wav", "*.mp3"])

    def _on_file_clicked(self, index):
        """Handle file item click"""
        file_path = self.file_model.filePath(index)
        if file_path and Path(file_path).is_file():
            self.file_selected.emit(file_path)
            self.signals.file_selected.emit(file_path)

    def _on_open_folder_clicked(self):
        """Handle open folder button click"""
        from PySide6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            str(Path.home()),
        )
        if folder:
            self.set_root_path(folder)

    def _on_external_file_selected(self, file_path: str):
        """Handle file selection from external source"""
        path = Path(file_path)
        if path.exists():
            # Navigate to the file's parent directory
            self.set_root_path(str(path.parent))

    def _refresh_files(self):
        """Refresh the file view"""
        current_root = self.file_model.rootPath()
        self.file_model.setRootPath("")
        self.file_model.setRootPath(current_root)

    def set_root_path(self, path: str):
        """Set the root path for the file browser"""
        if Path(path).exists():
            self.file_model.setRootPath(path)
            self.file_tree.setRootIndex(self.file_model.index(path))

    def get_selected_file(self) -> str | None:
        """Get the currently selected file path"""
        indexes = self.file_tree.selectedIndexes()
        if indexes:
            return self.file_model.filePath(indexes[0])
        return None
