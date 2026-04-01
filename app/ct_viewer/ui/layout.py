from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from . import mask_viz as ct_mask_viz
from . import widgets as ct_widgets
from .models import WINDOW_PRESETS


def build_ui(viewer: QMainWindow) -> None:
    central = QWidget()
    outer_layout = QHBoxLayout(central)
    outer_layout.setContentsMargins(18, 18, 18, 18)
    outer_layout.setSpacing(18)

    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.setChildrenCollapsible(False)

    splitter.addWidget(build_control_panel(viewer))
    splitter.addWidget(build_view_panel(viewer))
    splitter.setSizes([300, 1320])

    outer_layout.addWidget(splitter)
    viewer.setCentralWidget(central)

    status = QStatusBar()
    status.showMessage("Load a CT file or DICOM folder to begin.")
    viewer.setStatusBar(status)


def build_control_panel(viewer: QMainWindow) -> QWidget:
    panel = QFrame()
    panel.setObjectName("sidePanel")
    panel.setFixedWidth(340)
    panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    layout = QVBoxLayout(panel)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(0)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    scroll.setFixedWidth(304)
    scroll.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    content = QWidget()
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(16)

    title = QLabel("CT + Mask Viewer")
    title.setObjectName("panelTitle")

    subtitle = QLabel("Fast desktop viewer for CT volumes and accompanying segmentation masks.")
    subtitle.setWordWrap(True)
    subtitle.setObjectName("panelSubtitle")

    viewer.load_ct_button = QPushButton("Load CT File")
    viewer.load_dicom_button = QPushButton("Load DICOM Folder")
    viewer.add_masks_button = QPushButton("Add Mask Files")
    viewer.add_mask_folder_button = QPushButton("Add Mask Folder")
    viewer.reset_view_button = QPushButton("Reset Window + Slices")
    viewer.clear_masks_button = QPushButton("Clear Masks")

    for button in (
        viewer.load_ct_button,
        viewer.load_dicom_button,
        viewer.add_masks_button,
        viewer.add_mask_folder_button,
        viewer.reset_view_button,
        viewer.clear_masks_button,
    ):
        button.setMinimumHeight(42)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    viewer.volume_info_label = QLabel("No CT loaded")
    viewer.volume_info_label.setWordWrap(True)
    viewer.volume_info_label.setObjectName("infoBlock")
    viewer.volume_info_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    viewer.volume_info_label.setFixedHeight(118)

    viewer.window_preset_combo = QComboBox()
    viewer.window_preset_combo.addItems(list(WINDOW_PRESETS.keys()))

    viewer.window_center_slider = QSlider(Qt.Orientation.Horizontal)
    viewer.window_width_slider = QSlider(Qt.Orientation.Horizontal)
    viewer.overlay_opacity_slider = QSlider(Qt.Orientation.Horizontal)
    viewer.overlay_opacity_slider.setRange(0, 100)
    viewer.overlay_opacity_slider.setValue(45)

    viewer.window_center_value = QLabel("300")
    viewer.window_width_value = QLabel("1500")
    viewer.overlay_opacity_value = QLabel("45%")

    controls_form = QFormLayout()
    controls_form.setSpacing(10)
    controls_form.addRow("Preset", viewer.window_preset_combo)
    controls_form.addRow("Center", _build_slider_row(viewer.window_center_slider, viewer.window_center_value))
    controls_form.addRow("Width", _build_slider_row(viewer.window_width_slider, viewer.window_width_value))
    controls_form.addRow("Mask Opacity", _build_slider_row(viewer.overlay_opacity_slider, viewer.overlay_opacity_value))

    viewer.crosshair_checkbox = QCheckBox("Show crosshair")
    viewer.crosshair_checkbox.setChecked(True)

    viewer.mask_list = QListWidget()
    viewer.mask_list.setMinimumHeight(210)
    viewer.mask_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    hint = QLabel("Click inside any slice to jump the crosshair. Mouse wheel scrolls that plane.")
    hint.setWordWrap(True)
    hint.setObjectName("helperText")

    import_section = ct_widgets.CollapsibleSection("Import & Actions", expanded=False)
    import_section.content_layout.addWidget(viewer.load_ct_button)
    import_section.content_layout.addWidget(viewer.load_dicom_button)
    import_section.content_layout.addWidget(viewer.add_masks_button)
    import_section.content_layout.addWidget(viewer.add_mask_folder_button)
    import_section.content_layout.addWidget(viewer.reset_view_button)
    import_section.content_layout.addWidget(viewer.clear_masks_button)

    window_section = ct_widgets.CollapsibleSection("Windowing", expanded=False)
    windowing_layout = QVBoxLayout()
    windowing_layout.setSpacing(10)
    windowing_layout.addLayout(controls_form)
    windowing_layout.addWidget(viewer.crosshair_checkbox)
    window_section.content_layout.addLayout(windowing_layout)

    masks_section = ct_widgets.CollapsibleSection("Masks", expanded=True)
    masks_section.content_layout.addWidget(viewer.mask_list)
    masks_section.content_layout.addWidget(hint)

    content_layout.addWidget(title)
    content_layout.addWidget(subtitle)
    content_layout.addWidget(viewer.volume_info_label)
    content_layout.addWidget(import_section)
    content_layout.addWidget(window_section)
    content_layout.addWidget(masks_section, 1)
    content_layout.addStretch(1)

    scroll.setWidget(content)
    layout.addWidget(scroll, 1)

    viewer.load_ct_button.clicked.connect(viewer.select_ct_file)
    viewer.load_dicom_button.clicked.connect(viewer.select_dicom_folder)
    viewer.add_masks_button.clicked.connect(viewer.select_mask_files)
    viewer.add_mask_folder_button.clicked.connect(viewer.select_mask_folder)
    viewer.reset_view_button.clicked.connect(viewer.reset_view_state)
    viewer.clear_masks_button.clicked.connect(viewer.clear_masks)
    viewer.window_preset_combo.currentTextChanged.connect(viewer.apply_window_preset)
    viewer.window_center_slider.valueChanged.connect(viewer.on_window_slider_changed)
    viewer.window_width_slider.valueChanged.connect(viewer.on_window_slider_changed)
    viewer.overlay_opacity_slider.valueChanged.connect(viewer.on_overlay_opacity_changed)
    viewer.crosshair_checkbox.toggled.connect(lambda _: viewer.render_all_views())
    viewer.mask_list.itemChanged.connect(viewer.on_mask_item_changed)

    return panel


def build_view_panel(viewer: QMainWindow) -> QWidget:
    panel = QFrame()
    panel.setObjectName("viewPanel")

    layout = QVBoxLayout(panel)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(14)

    viewer.cursor_info_label = QLabel("Voxel: -, -, -    HU: -")
    viewer.cursor_info_label.setObjectName("cursorInfo")

    viewer.views = {
        "axial": ct_widgets.SliceView("axial"),
        "coronal": ct_widgets.SliceView("coronal"),
        "sagittal": ct_widgets.SliceView("sagittal"),
    }
    viewer.mask_viz = ct_mask_viz.MaskVisualizationPane()

    viewer.views["axial"].sliceChanged.connect(viewer.on_slice_changed)
    viewer.views["axial"].crosshairRequested.connect(viewer.on_view_clicked)

    axial_column = QSplitter(Qt.Orientation.Vertical)
    axial_column.setObjectName("maskAxialSplit")
    axial_column.setChildrenCollapsible(False)
    axial_column.setHandleWidth(0)
    axial_column.addWidget(viewer.mask_viz)
    axial_column.addWidget(viewer.views["axial"])
    axial_column.setSizes([560, 440])

    views_layout = QHBoxLayout()
    views_layout.setSpacing(14)
    views_layout.addWidget(axial_column, 1)
    for orientation in ("coronal", "sagittal"):
        view = viewer.views[orientation]
        view.sliceChanged.connect(viewer.on_slice_changed)
        view.crosshairRequested.connect(viewer.on_view_clicked)
        views_layout.addWidget(view, 1)

    layout.addWidget(viewer.cursor_info_label)
    layout.addLayout(views_layout, 1)
    return panel


def build_actions(viewer: QMainWindow) -> None:
    load_ct_action = QAction("Load CT File", viewer)
    load_ct_action.setShortcut(QKeySequence("Ctrl+O"))
    load_ct_action.triggered.connect(viewer.select_ct_file)

    load_dicom_action = QAction("Load DICOM Folder", viewer)
    load_dicom_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
    load_dicom_action.triggered.connect(viewer.select_dicom_folder)

    add_mask_action = QAction("Add Mask Files", viewer)
    add_mask_action.setShortcut(QKeySequence("Ctrl+M"))
    add_mask_action.triggered.connect(viewer.select_mask_files)

    add_folder_action = QAction("Add Mask Folder", viewer)
    add_folder_action.setShortcut(QKeySequence("Ctrl+Shift+M"))
    add_folder_action.triggered.connect(viewer.select_mask_folder)

    reset_action = QAction("Reset Window + Slices", viewer)
    reset_action.setShortcut(QKeySequence("R"))
    reset_action.triggered.connect(viewer.reset_view_state)

    clear_masks_action = QAction("Clear Masks", viewer)
    clear_masks_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
    clear_masks_action.triggered.connect(viewer.clear_masks)

    file_menu = viewer.menuBar().addMenu("File")
    file_menu.addAction(load_ct_action)
    file_menu.addAction(load_dicom_action)
    file_menu.addAction(add_mask_action)
    file_menu.addAction(add_folder_action)
    file_menu.addSeparator()
    file_menu.addAction(clear_masks_action)

    view_menu = viewer.menuBar().addMenu("View")
    view_menu.addAction(reset_action)

    viewer.addAction(load_ct_action)
    viewer.addAction(load_dicom_action)
    viewer.addAction(add_mask_action)
    viewer.addAction(add_folder_action)
    viewer.addAction(reset_action)
    viewer.addAction(clear_masks_action)


def _build_slider_row(slider: QSlider, value_label: QLabel) -> QWidget:
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(10)

    slider.setSingleStep(1)
    value_label.setMinimumWidth(56)
    value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    value_label.setStyleSheet("color: #e4edf5;")

    row.addWidget(slider, 1)
    row.addWidget(value_label)
    return container
