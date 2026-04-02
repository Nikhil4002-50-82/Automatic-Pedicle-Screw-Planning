from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QHeaderView,
    QTreeWidget,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QScrollArea,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from . import mask_viz as ct_mask_viz
from . import widgets as ct_widgets
from .models import WINDOW_PRESETS


def build_ui(viewer: QMainWindow) -> None:
    central = QWidget()
    outer_layout = QGridLayout(central)
    outer_layout.setContentsMargins(18, 18, 18, 18)
    outer_layout.setSpacing(18)

    content = QWidget()
    content_layout = QHBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(18)

    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.setChildrenCollapsible(False)

    splitter.addWidget(build_control_panel(viewer))
    splitter.addWidget(build_view_panel(viewer))
    splitter.setSizes([300, 1320])

    content_layout.addWidget(splitter)

    viewer.loading_overlay = _build_loading_overlay(viewer)
    viewer.loading_overlay.setVisible(False)

    outer_layout.addWidget(content, 0, 0)
    outer_layout.addWidget(viewer.loading_overlay, 0, 0)
    viewer.setCentralWidget(central)

    status = QStatusBar()
    status.showMessage("Load a CT file or DICOM folder to begin.")
    viewer.setStatusBar(status)


def _build_loading_overlay(viewer: QMainWindow) -> QWidget:
    overlay = QWidget()
    overlay.setObjectName("globalLoadingOverlay")
    overlay.setStyleSheet(
        """
        QWidget#globalLoadingOverlay {
            background: rgba(7, 12, 20, 0.72);
        }
        """
    )

    layout = QVBoxLayout(overlay)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(12)
    layout.addStretch(1)

    card = QFrame()
    card.setObjectName("loadingCard")
    card.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
    card.setMinimumSize(700, 270)
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(34, 28, 34, 28)
    card_layout.setSpacing(14)

    viewer.loading_overlay_title = QLabel("Loading masks")
    viewer.loading_overlay_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    viewer.loading_overlay_title.setObjectName("loadingOverlayTitle")
    viewer.loading_overlay_title.setStyleSheet("font-size: 19px;")

    viewer.loading_overlay_label = QLabel("Preparing mask overlays and 3D preview...")
    viewer.loading_overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    viewer.loading_overlay_label.setWordWrap(True)
    viewer.loading_overlay_label.setObjectName("loadingOverlayText")
    viewer.loading_overlay_label.setStyleSheet("font-size: 13px;")

    viewer.loading_overlay_bar = QProgressBar()
    viewer.loading_overlay_bar.setRange(0, 0)
    viewer.loading_overlay_bar.setTextVisible(False)
    viewer.loading_overlay_bar.setFixedHeight(10)
    viewer.loading_overlay_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    card_layout.addWidget(viewer.loading_overlay_title)
    card_layout.addWidget(viewer.loading_overlay_label)
    card_layout.addWidget(viewer.loading_overlay_bar)

    layout.addWidget(card, 0, Qt.AlignmentFlag.AlignCenter)
    layout.addStretch(1)
    return overlay


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

    viewer.load_ct_button = QPushButton("Load NIfTI")
    viewer.load_dicom_button = QPushButton("Load DICOM")
    viewer.add_masks_button = QPushButton("Add Mask(s)")
    viewer.add_masks_button.setObjectName("maskPopupButton")
    viewer.reset_view_button = QPushButton("Reset View")
    viewer.clear_masks_button = QPushButton("Clear Masks")

    for button in (
        viewer.load_ct_button,
        viewer.load_dicom_button,
        viewer.reset_view_button,
        viewer.clear_masks_button,
    ):
        button.setMinimumHeight(42)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    viewer.add_masks_button.setMinimumHeight(42)
    viewer.add_masks_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    viewer.volume_info_label = QLabel("No CT loaded")
    viewer.volume_info_label.setWordWrap(True)
    viewer.volume_info_label.setObjectName("infoBlock")
    viewer.volume_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    viewer.volume_info_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    viewer.volume_info_label.setMinimumHeight(74)

    viewer.study_tree = QTreeWidget()
    viewer.study_tree.setObjectName("studySwitcher")
    viewer.study_tree.setColumnCount(2)
    viewer.study_tree.setHeaderHidden(True)
    viewer.study_tree.header().setStretchLastSection(False)
    viewer.study_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    viewer.study_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    viewer.study_tree.setRootIsDecorated(True)
    viewer.study_tree.setIndentation(18)
    viewer.study_tree.setMinimumHeight(150)
    viewer.study_tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    viewer.study_tree.setUniformRowHeights(True)
    viewer.study_tree.setColumnWidth(1, 28)
    viewer.study_list = viewer.study_tree

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
    viewer.overlay_opacity_widget = _build_slider_row(viewer.overlay_opacity_slider, viewer.overlay_opacity_value)

    controls_form = QFormLayout()
    controls_form.setSpacing(10)
    controls_form.addRow("Preset", viewer.window_preset_combo)
    controls_form.addRow("Center", _build_slider_row(viewer.window_center_slider, viewer.window_center_value))
    controls_form.addRow("Width", _build_slider_row(viewer.window_width_slider, viewer.window_width_value))
    controls_form.addRow("Mask Opacity", viewer.overlay_opacity_widget)

    viewer.crosshair_checkbox = QCheckBox("Show crosshair")
    viewer.crosshair_checkbox.setChecked(True)

    hint = QLabel("Select a study to switch. Expand it to inspect child masks.")
    hint.setWordWrap(True)
    hint.setObjectName("helperText")

    import_section = ct_widgets.CollapsibleSection("Import & Actions", expanded=False)
    import_grid = QGridLayout()
    import_grid.setSpacing(10)
    import_grid.setContentsMargins(0, 0, 0, 0)
    import_grid.addWidget(viewer.load_ct_button, 0, 0)
    import_grid.addWidget(viewer.load_dicom_button, 0, 1)
    import_grid.addWidget(viewer.add_masks_button, 1, 0)
    import_grid.addWidget(viewer.clear_masks_button, 1, 1)
    import_grid.addWidget(viewer.reset_view_button, 2, 0, 1, 2)
    import_section.content_layout.addLayout(import_grid)

    viewer.study_section = ct_widgets.CollapsibleSection("Studies", expanded=True)
    viewer.study_section.content_layout.addWidget(viewer.study_tree)
    viewer.study_section.content_layout.addWidget(hint)

    window_section = ct_widgets.CollapsibleSection("Windowing", expanded=False)
    windowing_layout = QVBoxLayout()
    windowing_layout.setSpacing(10)
    windowing_layout.addLayout(controls_form)
    windowing_layout.addWidget(viewer.crosshair_checkbox)
    window_section.content_layout.addLayout(windowing_layout)

    content_layout.addWidget(title)
    content_layout.addWidget(subtitle)
    content_layout.addWidget(import_section)
    content_layout.addWidget(viewer.study_section)
    content_layout.addWidget(window_section)
    content_layout.addWidget(viewer.volume_info_label)
    content_layout.addStretch(1)

    scroll.setWidget(content)
    layout.addWidget(scroll, 1)

    viewer.load_ct_button.clicked.connect(viewer.select_ct_file)
    viewer.load_dicom_button.clicked.connect(viewer.select_dicom_folder)
    viewer.reset_view_button.clicked.connect(viewer.reset_view_state)
    viewer.clear_masks_button.clicked.connect(viewer.clear_masks)
    viewer.window_preset_combo.currentTextChanged.connect(viewer.apply_window_preset)
    viewer.window_center_slider.valueChanged.connect(viewer.on_window_slider_changed)
    viewer.window_width_slider.valueChanged.connect(viewer.on_window_slider_changed)
    viewer.overlay_opacity_slider.valueChanged.connect(viewer.on_overlay_opacity_changed)
    viewer.crosshair_checkbox.toggled.connect(lambda _: viewer.render_all_views())
    viewer.study_tree.itemSelectionChanged.connect(viewer.on_study_selected)
    viewer.study_tree.itemChanged.connect(viewer.on_mask_item_changed)

    mask_menu = QMenu(viewer)
    mask_menu.setObjectName("maskPopupMenu")
    mask_menu.addAction("Add Mask File(s)", viewer.select_mask_files)
    mask_menu.addAction("Add Mask Folder", viewer.select_mask_folder)
    viewer.add_masks_menu = mask_menu
    viewer.add_masks_button.clicked.connect(lambda: mask_menu.popup(viewer.add_masks_button.mapToGlobal(viewer.add_masks_button.rect().bottomLeft())))

    return panel


def build_view_panel(viewer: QMainWindow) -> QWidget:
    panel = QFrame()
    panel.setObjectName("viewPanel")

    layout = QVBoxLayout(panel)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(14)

    viewer.cursor_info_label = QLabel("Voxel: -, -, -    HU: -")
    viewer.cursor_info_label.setObjectName("cursorInfo")
    viewer._mask_visualizer_expanded = False

    viewer.views = {
        "axial": ct_widgets.SliceView("axial"),
        "coronal": ct_widgets.SliceView("coronal"),
        "sagittal": ct_widgets.SliceView("sagittal"),
    }
    viewer.mask_viz = ct_mask_viz.MaskVisualizationPane()
    viewer.mask_viz.doubleClicked.connect(viewer.toggle_mask_visualizer_expanded)
    viewer.mask_viz.set_opacity(1.0)

    viewer.views["axial"].sliceChanged.connect(viewer.on_slice_changed)
    viewer.views["axial"].crosshairRequested.connect(viewer.on_view_clicked)

    layout.addWidget(viewer.cursor_info_label)

    viewer.view_stack = QStackedWidget()
    viewer.normal_view_page = QWidget()
    viewer.expanded_view_page = QWidget()
    viewer.view_stack.addWidget(viewer.normal_view_page)
    viewer.view_stack.addWidget(viewer.expanded_view_page)

    _build_normal_view_page(viewer)
    _build_expanded_view_page(viewer)

    layout.addWidget(viewer.view_stack, 1)
    return panel


def _build_normal_view_page(viewer: QMainWindow) -> None:
    layout = QHBoxLayout(viewer.normal_view_page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(14)

    viewer.normal_axial_splitter = QSplitter(Qt.Orientation.Vertical)
    viewer.normal_axial_splitter.setObjectName("maskAxialSplit")
    viewer.normal_axial_splitter.setChildrenCollapsible(False)
    viewer.normal_axial_splitter.setHandleWidth(0)
    viewer.normal_axial_splitter.addWidget(viewer.mask_viz)
    viewer.normal_axial_splitter.addWidget(viewer.views["axial"])
    viewer.normal_axial_splitter.setSizes([560, 440])

    layout.addWidget(viewer.normal_axial_splitter, 1)
    for orientation in ("coronal", "sagittal"):
        view = viewer.views[orientation]
        view.sliceChanged.connect(viewer.on_slice_changed)
        view.crosshairRequested.connect(viewer.on_view_clicked)
        layout.addWidget(view, 1)


def _build_expanded_view_page(viewer: QMainWindow) -> None:
    layout = QVBoxLayout(viewer.expanded_view_page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)

    viewer.expanded_mask_host = QWidget()
    viewer.expanded_mask_host_layout = QVBoxLayout(viewer.expanded_mask_host)
    viewer.expanded_mask_host_layout.setContentsMargins(0, 0, 0, 0)
    viewer.expanded_mask_host_layout.setSpacing(12)
    viewer.expanded_mask_toolbar = QWidget()
    viewer.expanded_mask_toolbar_layout = QHBoxLayout(viewer.expanded_mask_toolbar)
    viewer.expanded_mask_toolbar_layout.setContentsMargins(0, 0, 0, 0)
    viewer.expanded_mask_toolbar_layout.setSpacing(10)

    viewer.expanded_return_button = QPushButton("Back to slices")
    viewer.expanded_return_button.setMinimumHeight(34)
    viewer.expanded_return_button.setFixedWidth(140)
    viewer.expanded_return_button.clicked.connect(viewer.toggle_mask_visualizer_expanded)

    viewer.expanded_hint_label = QLabel("Double-click the preview or press Esc to return to slices.")
    viewer.expanded_hint_label.setObjectName("helperText")
    viewer.expanded_hint_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    viewer.expanded_hint_label.setWordWrap(True)

    viewer.expanded_mask_hint = QLabel("Adjust 3D opacity below without re-rendering the mesh.")
    viewer.expanded_mask_hint.setObjectName("helperText")
    viewer.expanded_mask_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
    viewer.expanded_mask_hint.setWordWrap(True)

    viewer.expanded_overlay_opacity_slider = QSlider(Qt.Orientation.Horizontal)
    viewer.expanded_overlay_opacity_slider.setRange(0, 100)
    viewer.expanded_overlay_opacity_slider.setValue(int(round(viewer.mask_viz.opacity() * 100)))
    viewer.expanded_overlay_opacity_value = QLabel(f"{viewer.expanded_overlay_opacity_slider.value()}%")
    viewer.expanded_overlay_opacity_widget = _build_slider_row(
        viewer.expanded_overlay_opacity_slider,
        viewer.expanded_overlay_opacity_value,
    )
    viewer.expanded_overlay_opacity_slider.valueChanged.connect(viewer.on_expanded_overlay_opacity_changed)

    viewer.expanded_mask_toolbar_layout.addWidget(viewer.expanded_return_button, 0, Qt.AlignmentFlag.AlignLeft)
    viewer.expanded_mask_toolbar_layout.addWidget(viewer.expanded_hint_label, 1)

    viewer.expanded_mask_host_layout.addWidget(viewer.expanded_mask_toolbar)
    viewer.expanded_mask_host_layout.addWidget(viewer.expanded_mask_hint)
    viewer.expanded_mask_host_layout.addWidget(viewer.expanded_overlay_opacity_widget)

    layout.addWidget(viewer.expanded_mask_host, 1)

    shortcut = QShortcut(QKeySequence("Esc"), viewer)
    shortcut.activated.connect(viewer.toggle_mask_visualizer_expanded)


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
    viewer.recent_studies_menu = file_menu.addMenu("Recent Studies")
    viewer.recent_studies_menu.setObjectName("recentStudiesMenu")

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
