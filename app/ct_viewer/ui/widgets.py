from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QSlider, QToolButton, QVBoxLayout, QWidget

from .models import ORIENTATION_TITLES, clamp


class SliceCanvas(QLabel):
    imageClicked = pyqtSignal(str, int, int)
    wheelStepped = pyqtSignal(str, int)

    def __init__(self, orientation: str) -> None:
        super().__init__()
        self.orientation = orientation
        self._base_pixmap = QPixmap()
        self._logical_size = QSize()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(260, 260)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setText("Load a CT to start")
        self.setWordWrap(True)
        self.setStyleSheet(
            """
            QLabel {
                background: #0f1722;
                border: 1px solid #243244;
                border-radius: 14px;
                color: #9fb0c3;
                font-size: 13px;
            }
            """
        )

    def clear_image(self, message: str) -> None:
        self._base_pixmap = QPixmap()
        self._logical_size = QSize()
        self.clear()
        self.setText(message)

    def set_image(self, pixmap: QPixmap, logical_size: tuple[int, int]) -> None:
        self._base_pixmap = pixmap
        self._logical_size = QSize(logical_size[0], logical_size[1])
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_pixmap()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            coords = self._map_to_image(event.position().x(), event.position().y())
            if coords is not None:
                self.imageClicked.emit(self.orientation, coords[0], coords[1])
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta != 0:
            self.wheelStepped.emit(self.orientation, 1 if delta > 0 else -1)
        event.accept()

    def _refresh_pixmap(self) -> None:
        if self._base_pixmap.isNull():
            return

        scaled = self._base_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def _map_to_image(self, x_pos: float, y_pos: float) -> tuple[int, int] | None:
        shown = self.pixmap()
        if shown is None or shown.isNull() or self._logical_size.isEmpty():
            return None

        x_offset = (self.width() - shown.width()) / 2.0
        y_offset = (self.height() - shown.height()) / 2.0
        if x_pos < x_offset or y_pos < y_offset:
            return None
        if x_pos > x_offset + shown.width() or y_pos > y_offset + shown.height():
            return None

        logical_width = self._logical_size.width()
        logical_height = self._logical_size.height()
        image_x = int((x_pos - x_offset) * logical_width / max(shown.width(), 1))
        image_y = int((y_pos - y_offset) * logical_height / max(shown.height(), 1))
        image_x = clamp(image_x, 0, logical_width - 1)
        image_y = clamp(image_y, 0, logical_height - 1)
        return image_x, image_y


class SliceView(QWidget):
    sliceChanged = pyqtSignal(str, int)
    crosshairRequested = pyqtSignal(str, int, int)

    def __init__(self, orientation: str) -> None:
        super().__init__()
        self.orientation = orientation

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel(ORIENTATION_TITLES[orientation])
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #edf2f7;")

        self.canvas = SliceCanvas(orientation)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(lambda value: self.sliceChanged.emit(self.orientation, value))

        self.footer_label = QLabel("No volume loaded")
        self.footer_label.setStyleSheet("font-size: 12px; color: #92a4b8;")

        self.canvas.imageClicked.connect(self.crosshairRequested.emit)
        self.canvas.wheelStepped.connect(self._step_from_wheel)

        layout.addWidget(self.title_label)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.slider)
        layout.addWidget(self.footer_label)

    def clear_view(self) -> None:
        self.slider.blockSignals(True)
        self.slider.setRange(0, 0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.blockSignals(False)
        self.canvas.clear_image("Load a CT to start")
        self.footer_label.setText("No volume loaded")

    def set_slice_bounds(self, max_index: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setRange(0, max_index)
        self.slider.setValue(min(self.slider.value(), max_index))
        self.slider.setEnabled(True)
        self.slider.blockSignals(False)

    def set_slice_index(self, value: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)

    def set_image(self, pixmap: QPixmap, logical_size: tuple[int, int]) -> None:
        self.canvas.set_image(pixmap, logical_size)

    def set_footer(self, text: str) -> None:
        self.footer_label.setText(text)

    def _step_from_wheel(self, orientation: str, step: int) -> None:
        new_value = clamp(self.slider.value() + step, self.slider.minimum(), self.slider.maximum())
        if new_value != self.slider.value():
            self.slider.setValue(new_value)


class CollapsibleSection(QWidget):
    def __init__(self, title: str, expanded: bool = False) -> None:
        super().__init__()
        self.toggle_button = QToolButton()
        self.toggle_button.setObjectName("sectionToggle")
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toggle_button.toggled.connect(self._on_toggled)

        self.content_frame = QFrame()
        self.content_frame.setObjectName("collapsibleContent")
        self.content_frame.setVisible(expanded)
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setContentsMargins(0, 8, 0, 0)
        self.content_layout.setSpacing(10)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_frame)

    def _on_toggled(self, checked: bool) -> None:
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self.content_frame.setVisible(checked)

    def addWidget(self, widget: QWidget) -> None:
        self.content_layout.addWidget(widget)

    def addLayout(self, layout: QVBoxLayout) -> None:
        self.content_layout.addLayout(layout)
