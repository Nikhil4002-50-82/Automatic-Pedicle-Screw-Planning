from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ct_viewer.ui import layout as ct_layout  # noqa: E402
    from ct_viewer.ui import rendering as ct_rendering  # noqa: E402
    from ct_viewer.ui.focus import focus_indices_from_masks  # noqa: E402
    from ct_viewer.ui.models import (  # noqa: E402
        CTVolume,
        MASK_COLORS,
        MaskLayer,
        MaskLoadResult,
        ORIENTATION_TITLES,
        SLICE_AXES,
        WINDOW_PRESETS,
        clamp,
    )
    from ct_viewer.ui.runtime import install_qt_message_filter  # noqa: E402
    from ct_viewer.ui.workers import CTLoadWorker, MaskLoadWorker, MaskPreviewWorker  # noqa: E402
    from ct_viewer.ui.theme import apply_styles as apply_viewer_styles  # noqa: E402
else:
    from .ui import layout as ct_layout  # noqa: E402
    from .ui import rendering as ct_rendering  # noqa: E402
    from .ui.focus import focus_indices_from_masks  # noqa: E402
    from .ui.models import (  # noqa: E402
        CTVolume,
        MASK_COLORS,
        MaskLayer,
        MaskLoadResult,
        ORIENTATION_TITLES,
        SLICE_AXES,
        WINDOW_PRESETS,
        clamp,
    )
    from .ui.runtime import install_qt_message_filter  # noqa: E402
    from .ui.workers import CTLoadWorker, MaskLoadWorker, MaskPreviewWorker  # noqa: E402
    from .ui.theme import apply_styles as apply_viewer_styles  # noqa: E402


class CTMaskViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        install_qt_message_filter()
        self.ct_volume: CTVolume | None = None
        self.ct_zooms = (1.0, 1.0, 1.0)
        self.mask_layers: list[MaskLayer] = []
        self.current_indices = [0, 0, 0]
        self.auto_window = (300, 1500)
        self.ct_intensity_summary = "Intensity range: unavailable"
        self.ct_slice_cache: dict[tuple[str, int], np.ndarray] = {}
        self.mask_slice_cache: dict[tuple[int, str, int], np.ndarray] = {}
        self._ct_thread: QThread | None = None
        self._mask_thread: QThread | None = None
        self._ct_worker: CTLoadWorker | None = None
        self._mask_worker: MaskLoadWorker | None = None
        self._mask_preview_thread: QThread | None = None
        self._mask_preview_worker: MaskPreviewWorker | None = None
        self._mask_preview_generation = 0
        self._mask_preview_needs_refresh = False
        self._window_control_lock = False

        self.setWindowTitle("CT and Segmentation Viewer")
        self.resize(1680, 980)

        ct_layout.build_ui(self)
        ct_layout.build_actions(self)
        apply_viewer_styles(self)
        self.clear_viewer()

    def clear_viewer(self) -> None:
        if getattr(self, "_mask_visualizer_expanded", False):
            self._set_mask_visualizer_expanded(False)
        self.hide_loading_overlay()
        self.ct_volume = None
        self.mask_layers = []
        self.ct_slice_cache.clear()
        self.mask_slice_cache.clear()
        self.volume_info_label.setText("No CT loaded")
        self.cursor_info_label.setText("Voxel: -, -, -    HU: -")
        for view in self.views.values():
            view.clear_view()
        if hasattr(self, "mask_viz"):
            self.mask_viz.clear_view()
        self._mask_preview_needs_refresh = False
        self.mask_list.clear()
        self.window_center_slider.setEnabled(False)
        self.window_width_slider.setEnabled(False)
        self.overlay_opacity_slider.setEnabled(False)
        self.crosshair_checkbox.setEnabled(False)
        self.window_preset_combo.setEnabled(False)
        self._set_loading_state("Load a CT file or DICOM folder to begin.", busy=False)

    def toggle_mask_visualizer_expanded(self) -> None:
        self._set_mask_visualizer_expanded(not getattr(self, "_mask_visualizer_expanded", False))

    def _set_mask_visualizer_expanded(self, expanded: bool) -> None:
        if not hasattr(self, "view_stack") or not hasattr(self, "mask_viz"):
            return
        if getattr(self, "_mask_visualizer_expanded", False) == expanded:
            return

        self._mask_visualizer_expanded = expanded

        if expanded:
            self.mask_viz.setParent(None)
            self.expanded_mask_host_layout.insertWidget(0, self.mask_viz, 1)
            self.view_stack.setCurrentWidget(self.expanded_view_page)
            self.expanded_overlay_opacity_slider.blockSignals(True)
            self.expanded_overlay_opacity_slider.setValue(int(round(self.mask_viz.opacity() * 100)))
            self.expanded_overlay_opacity_slider.blockSignals(False)
            self.expanded_overlay_opacity_value.setText(f"{self.expanded_overlay_opacity_slider.value()}%")
        else:
            self.mask_viz.setParent(None)
            self.normal_axial_splitter.insertWidget(0, self.mask_viz)
            self.normal_axial_splitter.setSizes([560, 440])
            self.view_stack.setCurrentWidget(self.normal_view_page)

        self.render_all_views()
        QTimer.singleShot(0, self.mask_viz.notify_resize)

    def select_ct_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CT Volume",
            "",
            "CT Volumes (*.nii *.nii.gz *.dcm);;All Files (*.*)",
        )
        if path:
            self.load_ct_async(path)

    def select_dicom_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            self.load_ct_async(folder)

    def select_mask_files(self) -> None:
        if not self._require_ct_loaded():
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Segmentation Masks",
            "",
            "NIfTI Files (*.nii *.nii.gz)",
        )
        if paths:
            self.load_masks_async(paths)

    def select_mask_folder(self) -> None:
        if not self._require_ct_loaded():
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Mask Folder")
        if not folder:
            return

        folder_path = Path(folder)
        mask_paths = sorted(str(path) for path in folder_path.glob("*.nii"))
        mask_paths.extend(sorted(str(path) for path in folder_path.glob("*.nii.gz")))
        if not mask_paths:
            QMessageBox.information(self, "No Masks Found", "The selected folder does not contain any NIfTI masks.")
            return

        self.load_masks_async(mask_paths)

    def load_ct_async(self, path: str) -> None:
        if self._ct_thread is not None:
            return

        self._set_loading_state("Loading CT...", busy=True)
        thread = QThread(self)
        worker = CTLoadWorker(path)
        self._ct_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_ct_loaded)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_ct_thread)
        self._ct_thread = thread
        thread.start()

    def load_masks_async(self, paths: list[str]) -> None:
        if self.ct_volume is None or self._mask_thread is not None:
            return

        seen_paths = {os.path.normcase(os.path.abspath(layer.path)) for layer in self.mask_layers}
        unique_paths = [path for path in paths if os.path.normcase(os.path.abspath(path)) not in seen_paths]
        if not unique_paths:
            self.statusBar().showMessage("All selected masks are already loaded.", 4000)
            return

        self.show_loading_overlay("Loading masks and preparing overlays...")
        self._set_loading_state("Loading masks...", busy=True)
        thread = QThread(self)
        worker = MaskLoadWorker(unique_paths, self.ct_volume.image, len(self.mask_layers))
        self._mask_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_masks_loaded)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_mask_thread)
        self._mask_thread = thread
        thread.start()

    def clear_masks(self) -> None:
        if not self.mask_layers:
            return

        self.mask_layers = []
        self.mask_slice_cache.clear()
        self.refresh_mask_list()
        self.update_volume_info()
        self.render_all_views()
        self.refresh_mask_visualization()
        self.statusBar().showMessage("Cleared all masks.", 4000)

    def _on_ct_loaded(self, volume: CTVolume) -> None:
        self.ct_volume = volume
        self.ct_zooms = volume.zooms
        self.mask_layers = []
        self.ct_slice_cache.clear()
        self.mask_slice_cache.clear()
        self.current_indices = [dimension // 2 for dimension in volume.shape]
        self.ct_intensity_summary = (
            f"Intensity range: {volume.summary.minimum:.1f} to {volume.summary.maximum:.1f}"
        )

        self._configure_window_controls()
        self._configure_slice_views()
        self.refresh_mask_list()
        self.update_volume_info()
        self.render_all_views()
        self.refresh_mask_visualization()
        self._set_loading_state(f"Loaded CT: {Path(volume.path).name}", busy=False)

        if volume.summary.is_constant:
            QMessageBox.warning(
                self,
                "Volume Appears Empty",
                (
                    f"{Path(volume.path).name} has no visible intensity variation.\n\n"
                    "The loaded volume is constant, so the slice views will look blank.\n"
                    "If this is a segmentation export, load the original CT instead and use this file as a mask."
                ),
            )

    def _on_masks_loaded(self, result: MaskLoadResult) -> None:
        if result.layers:
            self.mask_layers.extend(result.layers)
            self.mask_slice_cache.clear()
            focused_indices = focus_indices_from_masks(self.ct_volume.shape, result.layers) if self.ct_volume is not None else None
            if focused_indices is not None:
                self.current_indices = focused_indices
            self.refresh_mask_list()
            self.update_volume_info()
            self.render_all_views()
            self.refresh_mask_visualization()
            self._set_loading_state(f"Loaded {len(result.layers)} mask(s).", busy=False)
        else:
            self._set_loading_state("No masks were loaded.", busy=False)
            self.hide_loading_overlay()

        if result.warnings:
            QMessageBox.information(self, "Mask Load Notes", "\n".join(result.warnings))

    def _on_worker_failed(self, title: str, message: str) -> None:
        self._set_loading_state("Load failed.", busy=False)
        self.hide_loading_overlay()
        self._show_error(title, message)

    def _clear_ct_thread(self) -> None:
        self._ct_thread = None
        self._ct_worker = None
        self._set_loading_state(self.statusBar().currentMessage(), busy=self._mask_thread is not None)

    def _clear_mask_thread(self) -> None:
        self._mask_thread = None
        self._mask_worker = None
        self._set_loading_state(self.statusBar().currentMessage(), busy=self._ct_thread is not None)

    def _set_loading_state(self, message: str, busy: bool) -> None:
        self.load_ct_button.setEnabled(not busy and self._ct_thread is None)
        self.load_dicom_button.setEnabled(not busy and self._ct_thread is None)
        self.add_masks_button.setEnabled(not busy and self.ct_volume is not None and self._mask_thread is None)
        self.add_mask_folder_button.setEnabled(not busy and self.ct_volume is not None and self._mask_thread is None)
        self.reset_view_button.setEnabled(not busy and self.ct_volume is not None)
        self.clear_masks_button.setEnabled(not busy and bool(self.mask_layers))
        self.statusBar().showMessage(message)

    def reset_view_state(self) -> None:
        if self.ct_volume is None:
            return

        self.current_indices = [dimension // 2 for dimension in self.ct_volume.shape]
        self.ct_slice_cache.clear()
        self.mask_slice_cache.clear()
        self.apply_window_preset("Auto")
        self.render_all_views()
        self.statusBar().showMessage("Reset slices and windowing.", 4000)

    def _configure_window_controls(self) -> None:
        if self.ct_volume is None:
            return

        summary = self.ct_volume.summary
        data_min = int(np.floor(summary.minimum))
        data_max = int(np.ceil(summary.maximum))
        slider_center_min = min(data_min, -1500)
        slider_center_max = max(data_max, 3000)
        slider_width_max = max(slider_center_max - slider_center_min, 4000)

        self.auto_window = (
            int(round((summary.low_percentile + summary.high_percentile) / 2.0)),
            int(round(max(summary.high_percentile - summary.low_percentile, 1.0))),
        )

        self.window_center_slider.setRange(slider_center_min, slider_center_max)
        self.window_width_slider.setRange(1, slider_width_max)
        self.window_center_slider.setEnabled(True)
        self.window_width_slider.setEnabled(True)
        self.overlay_opacity_slider.setEnabled(True)
        self.crosshair_checkbox.setEnabled(True)
        self.window_preset_combo.setEnabled(True)
        self.apply_window_preset("Auto")

    def _configure_slice_views(self) -> None:
        if self.ct_volume is None:
            return

        for orientation, view in self.views.items():
            axis = SLICE_AXES[orientation]
            view.set_slice_bounds(self.ct_volume.shape[axis] - 1)
            view.set_slice_index(self.current_indices[axis])

    def update_volume_info(self) -> None:
        if self.ct_volume is None:
            self.volume_info_label.setText("No CT loaded")
            return

        shape_text = " x ".join(str(value) for value in self.ct_volume.shape)
        spacing_text = " x ".join(f"{value:.2f} mm" for value in self.ct_zooms)
        mask_count = len(self.mask_layers)
        filename = Path(self.ct_volume.path).name
        metrics = QFontMetrics(self.volume_info_label.font())
        elided_name = metrics.elidedText(filename, Qt.TextElideMode.ElideMiddle, 240)

        self.volume_info_label.setText(
            "\n".join(
                [
                    f"CT: {elided_name}",
                    f"Shape: {shape_text}",
                    f"Spacing: {spacing_text}",
                    f"{self.ct_intensity_summary} | Masks: {mask_count} | Orientation: canonical RAS+",
                ]
            )
        )

    def refresh_mask_list(self) -> None:
        self.mask_list.blockSignals(True)
        self.mask_list.clear()

        for layer in self.mask_layers:
            item = QListWidgetItem(layer.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
            item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
            item.setForeground(QColor(*layer.color))
            tooltip = layer.path
            if layer.voxel_count is not None:
                tooltip = f"{tooltip}\nVoxels: {layer.voxel_count:,}"
            item.setToolTip(tooltip)
            self.mask_list.addItem(item)

        self.mask_list.blockSignals(False)

    def on_mask_item_changed(self, item: QListWidgetItem) -> None:
        row = self.mask_list.row(item)
        if row < 0 or row >= len(self.mask_layers):
            return

        self.mask_layers[row].visible = item.checkState() == Qt.CheckState.Checked
        self.mask_slice_cache.clear()
        self.render_all_views()
        self.refresh_mask_visualization()

    def apply_window_preset(self, preset_name: str) -> None:
        if self.ct_volume is None or self._window_control_lock:
            return
        if preset_name == "Custom":
            return

        if preset_name == "Auto":
            center, width = self.auto_window
        else:
            preset = WINDOW_PRESETS.get(preset_name)
            if preset is None:
                return
            center, width = preset

        self._window_control_lock = True
        if self.window_preset_combo.currentText() != preset_name:
            self.window_preset_combo.setCurrentText(preset_name)
        self.window_center_slider.setValue(clamp(center, self.window_center_slider.minimum(), self.window_center_slider.maximum()))
        self.window_width_slider.setValue(clamp(width, self.window_width_slider.minimum(), self.window_width_slider.maximum()))
        self.window_center_value.setText(str(self.window_center_slider.value()))
        self.window_width_value.setText(str(self.window_width_slider.value()))
        self._window_control_lock = False
        self.render_all_views()

    def on_window_slider_changed(self) -> None:
        if self.ct_volume is None:
            return

        self.window_center_value.setText(str(self.window_center_slider.value()))
        self.window_width_value.setText(str(self.window_width_slider.value()))

        if not self._window_control_lock and self.window_preset_combo.currentText() != "Custom":
            self._window_control_lock = True
            self.window_preset_combo.setCurrentText("Custom")
            self._window_control_lock = False

        self.render_all_views()

    def on_overlay_opacity_changed(self, value: int) -> None:
        self.overlay_opacity_value.setText(f"{value}%")
        self.render_all_views()

    def on_expanded_overlay_opacity_changed(self, value: int) -> None:
        self.expanded_overlay_opacity_value.setText(f"{value}%")
        self.mask_viz.set_opacity(value / 100.0)

    def on_slice_changed(self, orientation: str, index: int) -> None:
        if self.ct_volume is None:
            return

        self.current_indices[SLICE_AXES[orientation]] = index
        self.render_all_views()

    def on_view_clicked(self, orientation: str, image_x: int, image_y: int) -> None:
        if self.ct_volume is None:
            return

        nx, ny, nz = self.ct_volume.shape
        if orientation == "axial":
            self.current_indices[0] = clamp(image_x, 0, nx - 1)
            self.current_indices[1] = clamp(ny - 1 - image_y, 0, ny - 1)
        elif orientation == "coronal":
            self.current_indices[0] = clamp(image_x, 0, nx - 1)
            self.current_indices[2] = clamp(nz - 1 - image_y, 0, nz - 1)
        elif orientation == "sagittal":
            self.current_indices[1] = clamp(ny - 1 - image_x, 0, ny - 1)
            self.current_indices[2] = clamp(nz - 1 - image_y, 0, nz - 1)

        self.render_all_views()

    def render_all_views(self) -> None:
        if self.ct_volume is None:
            return

        center = self.window_center_slider.value()
        width = self.window_width_slider.value()
        if getattr(self, "_mask_visualizer_expanded", False) and hasattr(self, "expanded_overlay_opacity_slider"):
            opacity = self.expanded_overlay_opacity_slider.value() / 100.0
        else:
            opacity = self.overlay_opacity_slider.value() / 100.0

        for orientation, view in self.views.items():
            pixmap, logical_size = self._render_view(orientation, center, width, opacity)
            axis = SLICE_AXES[orientation]
            slice_index = self.current_indices[axis]
            view.set_slice_index(slice_index)
            view.set_image(pixmap, logical_size)
            view.set_footer(f"Slice {slice_index + 1} / {self.ct_volume.shape[axis]}")

        self._update_cursor_info()

    def refresh_mask_visualization(self) -> None:
        if not hasattr(self, "mask_viz"):
            return

        if self.ct_volume is None:
            self.mask_viz.clear_view()
            self.hide_loading_overlay()
            return

        visible_layers = [layer for layer in self.mask_layers if layer.visible]
        if not visible_layers:
            self.mask_viz.clear_view()
            self.hide_loading_overlay()
            return

        if self._mask_preview_thread is not None:
            self._mask_preview_needs_refresh = True
            return

        self._mask_preview_generation += 1
        generation = self._mask_preview_generation
        self.show_loading_overlay(f"Rendering {len(visible_layers)} visible mask(s)...")
        self.mask_viz.set_busy(f"Rendering {len(visible_layers)} visible mask(s)...")

        thread = QThread(self)
        worker = MaskPreviewWorker(generation, self.ct_volume, visible_layers)
        self._mask_preview_thread = thread
        self._mask_preview_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_mask_preview_ready)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_mask_preview_thread)
        thread.start()

    def _on_mask_preview_ready(self, generation: int, figure_json: object) -> None:
        if generation != self._mask_preview_generation or not hasattr(self, "mask_viz"):
            return

        self.mask_viz.set_data_json(figure_json if isinstance(figure_json, dict) else None, len([layer for layer in self.mask_layers if layer.visible]))
        self._mask_preview_needs_refresh = False

    def _clear_mask_preview_thread(self) -> None:
        self._mask_preview_thread = None
        self._mask_preview_worker = None
        if self._mask_preview_needs_refresh and self.ct_volume is not None and any(layer.visible for layer in self.mask_layers):
            self._mask_preview_needs_refresh = False
            QTimer.singleShot(0, self.refresh_mask_visualization)
            return
        QTimer.singleShot(120, self.hide_loading_overlay)

    def _render_view(self, orientation: str, center: int, width: int, opacity: float) -> tuple[QPixmap, tuple[int, int]]:
        ct_slice = self._get_ct_slice(orientation)
        rgba = ct_rendering.grayscale_rgba(ct_slice, center, width)

        for layer_index, layer in enumerate(self.mask_layers):
            if not layer.visible:
                continue
            mask_slice = self._get_mask_slice(layer_index, layer, orientation)
            ct_rendering.blend_mask(rgba, mask_slice, layer.color, opacity)

        image = ct_rendering.qimage_from_rgba(rgba)
        if self.crosshair_checkbox.isChecked():
            cross_x, cross_y = ct_rendering.crosshair_position(self.ct_volume.shape, orientation, self.current_indices)
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            pen = QPen(QColor("#44d7ff"))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(cross_x, 0, cross_x, image.height() - 1)
            painter.drawLine(0, cross_y, image.width() - 1, cross_y)
            left_label, right_label, top_label, bottom_label = ct_rendering.orientation_labels(orientation)
            painter.setPen(QPen(QColor("#f7fbff")))
            painter.drawText(12, image.height() // 2, left_label)
            painter.drawText(image.width() - 22, image.height() // 2, right_label)
            painter.drawText(image.width() // 2 - 6, 20, top_label)
            painter.drawText(image.width() // 2 - 6, image.height() - 12, bottom_label)
            painter.end()

        pixmap = QPixmap.fromImage(image)
        width_spacing, height_spacing = ct_rendering.display_spacing(self.ct_zooms, orientation)
        target_width, target_height = ct_rendering.physical_display_size(image.width(), image.height(), width_spacing, height_spacing)
        if target_width != image.width() or target_height != image.height():
            pixmap = pixmap.scaled(
                target_width,
                target_height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )

        return pixmap, (image.width(), image.height())

    def _get_ct_slice(self, orientation: str) -> np.ndarray:
        axis = SLICE_AXES[orientation]
        index = self.current_indices[axis]
        cache_key = (orientation, index)
        cached = self.ct_slice_cache.get(cache_key)
        if cached is not None:
            return cached

        ct_slice = ct_rendering.extract_slice(self.ct_volume.image, orientation, self.current_indices)
        if not np.isfinite(ct_slice).all():
            ct_slice = np.nan_to_num(ct_slice, copy=False)

        self.ct_slice_cache = {cache_key: ct_slice}
        return ct_slice

    def _get_mask_slice(self, layer_index: int, layer: MaskLayer, orientation: str) -> np.ndarray:
        axis = SLICE_AXES[orientation]
        index = self.current_indices[axis]
        cache_key = (layer_index, orientation, index)
        cached = self.mask_slice_cache.get(cache_key)
        if cached is not None:
            return cached

        mask_slice = ct_rendering.extract_slice(layer.image, orientation, self.current_indices) != 0
        if len(self.mask_slice_cache) > 18:
            self.mask_slice_cache.clear()
        self.mask_slice_cache[cache_key] = mask_slice
        return mask_slice

    def _update_cursor_info(self) -> None:
        if self.ct_volume is None:
            self.cursor_info_label.setText("Voxel: -, -, -    HU: -")
            return

        x_idx, y_idx, z_idx = self.current_indices
        hu_value = float(np.asarray(self.ct_volume.image.dataobj[x_idx, y_idx, z_idx], dtype=np.float32))
        active_masks = []
        for layer_index, layer in enumerate(self.mask_layers):
            if not layer.visible:
                continue
            voxel_value = np.asarray(layer.image.dataobj[x_idx, y_idx, z_idx])
            if np.any(voxel_value != 0):
                active_masks.append(layer.name)
        mask_text = ", ".join(active_masks) if active_masks else "None"
        self.cursor_info_label.setText(
            f"Voxel: x={x_idx}  y={y_idx}  z={z_idx}    HU: {hu_value:.1f}    Masks here: {mask_text}"
        )

    def _require_ct_loaded(self) -> bool:
        if self.ct_volume is not None:
            return True
        QMessageBox.information(self, "Load CT First", "Load a CT volume before adding segmentation masks.")
        return False

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def show_loading_overlay(self, message: str) -> None:
        if not hasattr(self, "loading_overlay"):
            return
        self.loading_overlay_label.setText(message)
        self.loading_overlay.setVisible(True)
        self.loading_overlay.raise_()

    def hide_loading_overlay(self) -> None:
        if not hasattr(self, "loading_overlay"):
            return
        self.loading_overlay.setVisible(False)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast CT and segmentation viewer")
    parser.add_argument(
        "source",
        nargs="?",
        help="Optional CT input path. Accepts a NIfTI file or a DICOM folder.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])
    install_qt_message_filter()
    app = QApplication([sys.argv[0]])
    app.setApplicationName("CT and Segmentation Viewer")
    viewer = CTMaskViewer()
    viewer.show()

    if args.source:
        QTimer.singleShot(0, lambda: viewer.load_ct_async(args.source))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
