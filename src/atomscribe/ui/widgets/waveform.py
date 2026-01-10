"""Audio waveform visualization widget"""

import random
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

from ...styles.colors import NotionColors


class WaveformWidget(QWidget):
    """
    Real-time audio waveform visualizer.
    Displays animated bars representing audio levels.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("waveformWidget")
        self.setMinimumHeight(40)
        self.setMinimumWidth(200)

        # Waveform data
        self._num_bars = 32
        self._levels = [0.0] * self._num_bars
        self._target_levels = [0.0] * self._num_bars
        self._is_active = False

        # Colors
        self._active_color = QColor(NotionColors.WAVEFORM_ACTIVE)
        self._inactive_color = QColor(NotionColors.WAVEFORM_INACTIVE)

        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_animation)
        self._timer.start(50)  # 20 FPS

    def set_level(self, level: float):
        """
        Set the current audio level (0.0 - 1.0).
        This will animate the waveform bars.
        """
        level = max(0.0, min(1.0, level))

        if self._is_active:
            # Generate random variation based on level
            for i in range(self._num_bars):
                # Create wave-like pattern with some randomness
                base = level * (0.3 + 0.7 * random.random())
                # Add position-based variation
                wave_factor = abs((i - self._num_bars / 2) / (self._num_bars / 2))
                self._target_levels[i] = base * (1.0 - wave_factor * 0.5)

    def set_active(self, active: bool):
        """Set whether the waveform is actively recording"""
        self._is_active = active
        if not active:
            # Fade out all bars
            self._target_levels = [0.0] * self._num_bars

    def _update_animation(self):
        """Smooth animation update"""
        changed = False
        for i in range(self._num_bars):
            diff = self._target_levels[i] - self._levels[i]
            if abs(diff) > 0.01:
                # Smooth interpolation
                self._levels[i] += diff * 0.3
                changed = True
            else:
                self._levels[i] = self._target_levels[i]

        if changed:
            self.update()

        # If active, continuously generate new target levels
        if self._is_active:
            self.set_level(0.5 + 0.5 * random.random())

    def paintEvent(self, event):
        """Draw the waveform bars"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        bg_color = QColor(NotionColors.BACKGROUND_TERTIARY)
        painter.fillRect(self.rect(), bg_color)

        # Calculate bar dimensions
        width = self.width()
        height = self.height()
        bar_width = (width - (self._num_bars + 1) * 2) / self._num_bars
        max_bar_height = height - 8

        # Draw bars
        color = self._active_color if self._is_active else self._inactive_color
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))

        for i, level in enumerate(self._levels):
            x = 2 + i * (bar_width + 2)
            bar_height = max(4, level * max_bar_height)
            y = (height - bar_height) / 2

            # Draw rounded bar
            rect = QRectF(x, y, bar_width, bar_height)
            painter.drawRoundedRect(rect, 2, 2)

        painter.end()

    def start_demo(self):
        """Start demo animation for UI preview"""
        self._is_active = True
        self.set_level(0.5)

    def stop_demo(self):
        """Stop demo animation"""
        self._is_active = False
