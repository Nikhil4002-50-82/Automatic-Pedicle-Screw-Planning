from __future__ import annotations

import json
import tempfile
from typing import Iterable
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget

from .models import CTVolume, MaskLayer

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - optional dependency
    QWebEngineView = None

try:
    from ...visualizer_unified import (
        _ensure_plotly_imports,
        _ensure_plotlyjs_bundle,
        _figure_without_embedded_controls,
        _style_config,
    )
except ImportError:  # pragma: no cover - script-style fallback
    from visualizer_unified import (
        _ensure_plotly_imports,
        _ensure_plotlyjs_bundle,
        _figure_without_embedded_controls,
        _style_config,
    )

_measure = None


def _ensure_measure():
    global _measure
    if _measure is None:
        from skimage import measure

        _measure = measure
    return _measure


def _rgb_string(color: tuple[int, int, int]) -> str:
    return f"rgb({color[0]}, {color[1]}, {color[2]})"


def _marching_step(shape: tuple[int, int, int]) -> int:
    largest = max(shape)
    if largest <= 128:
        return 1
    if largest <= 256:
        return 2
    if largest <= 512:
        return 3
    return 4


def _mask_trace(mask: np.ndarray, layer: MaskLayer):
    go, _ = _ensure_plotly_imports()
    measure = _ensure_measure()

    mask = np.asarray(mask)
    if mask.ndim != 3 or not np.any(mask):
        return None

    padded = np.pad(mask.astype(np.float32, copy=False), 1, mode="constant")
    try:
        verts, faces, _, _ = measure.marching_cubes(
            padded,
            level=0.5,
            step_size=_marching_step(mask.shape),
        )
    except ValueError:
        return None

    verts -= 1.0
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=_rgb_string(layer.color),
        opacity=0.62,
        name=layer.name,
        flatshading=False,
        lighting=dict(ambient=0.58, diffuse=0.72, specular=0.18, roughness=0.48, fresnel=0.08),
        lightposition=dict(x=140, y=140, z=120),
        hoverinfo="skip",
        showscale=False,
    )


def _mask_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0).astype(np.float32)
    maxs = coords.max(axis=0).astype(np.float32)
    return mins, maxs


def build_mask_preview_figure(ct_volume: CTVolume | None, layers: Iterable[MaskLayer]):
    go, _ = _ensure_plotly_imports()
    style = _style_config("cinematic", "dark")
    fig = go.Figure()

    visible_layers = [layer for layer in layers if layer.visible]
    bounds: list[tuple[np.ndarray, np.ndarray]] = []
    for layer in visible_layers:
        mask = np.asarray(layer.image.dataobj)
        mask_bounds = _mask_bounds(mask)
        if mask_bounds is not None:
            bounds.append(mask_bounds)
        trace = _mask_trace(mask, layer)
        if trace is not None:
            fig.add_trace(trace)

    if not fig.data:
        fig.add_annotation(
            text="Load masks to preview them in 3D",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#9fb1c4", size=15),
        )

    if bounds:
        mins = np.min(np.stack([item[0] for item in bounds], axis=0), axis=0)
        maxs = np.max(np.stack([item[1] for item in bounds], axis=0), axis=0)
        spans = np.maximum(maxs - mins, 1.0)
        padding = np.maximum(spans * 0.20, 8.0)
        x_range = [float(mins[0] - padding[0]), float(maxs[0] + padding[0])]
        y_range = [float(mins[1] - padding[1]), float(maxs[1] + padding[1])]
        z_range = [float(mins[2] - padding[2]), float(maxs[2] + padding[2])]
    elif ct_volume is not None:
        x_range = [0, ct_volume.shape[0] - 1]
        y_range = [0, ct_volume.shape[1] - 1]
        z_range = [0, ct_volume.shape[2] - 1]
    else:
        x_range = y_range = z_range = None

    fig.update_layout(
        template=style["template"],
        paper_bgcolor=style["paper_bgcolor"],
        plot_bgcolor=style["plot_bgcolor"],
        margin=dict(l=0, r=0, t=16, b=0),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(16, 25, 38, 0.84)",
            bordercolor="rgba(151, 164, 180, 0.22)",
            borderwidth=1,
            font=dict(color="#f2f7fc", size=11),
            orientation="h",
            x=0.0,
            y=1.01,
        ),
        scene=dict(
            aspectmode="data",
            bgcolor=style["scene_bgcolor"],
            xaxis=dict(visible=False, range=x_range),
            yaxis=dict(visible=False, range=y_range),
            zaxis=dict(visible=False, range=z_range),
            camera=dict(eye=dict(x=1.15, y=1.1, z=0.8)),
        ),
        uirevision="mask-preview",
    )
    return fig


def _figure_to_html_document() -> str:
    _, pio = _ensure_plotly_imports()
    bundle_uri = Path(_ensure_plotlyjs_bundle()).as_uri()
    style = _style_config("cinematic", "dark")
    paper_bg = style["paper_bgcolor"]
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="{bundle_uri}"></script>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: {paper_bg};
    }}
    body {{
      font-family: Segoe UI, sans-serif;
    }}
    .plotly-graph-div {{
      width: 100% !important;
      height: 100% !important;
    }}
    .modebar-btn,
    g.updatemenu-button,
    g.updatemenu-button *,
    g.slider *,
    .legendtoggle {{
      cursor: pointer !important;
    }}
    .modebar {{
      background: rgba(8, 14, 24, 0.44) !important;
      border: 1px solid rgba(148, 163, 184, 0.14);
      border-radius: 10px;
      padding: 4px;
      backdrop-filter: blur(6px);
    }}
    .modebar-btn {{
      border-radius: 8px !important;
      transition: background-color 110ms ease, transform 110ms ease, opacity 110ms ease;
    }}
    .modebar-btn:hover {{
      background: rgba(91, 243, 255, 0.18) !important;
      transform: translateY(-1px);
    }}
    .modebar-btn.active {{
      background: rgba(91, 243, 255, 0.26) !important;
    }}
    #plot {{
      width: 100%;
      height: calc(100% - 0px);
    }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    window.renderMaskPreview = function(figJson) {{
      const fig = JSON.parse(figJson);
      const plot = document.getElementById('plot');
      if (!window.Plotly || !plot) return;
      Plotly.react(plot, fig.data || [], fig.layout || {{}}, {{
        displayModeBar: true,
        displaylogo: false,
        scrollZoom: true
      }});
    }};
  </script>
</body>
</html>
"""


def _write_html_document() -> str:
    html_document = _figure_to_html_document()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
        tmpfile.write(html_document.encode("utf-8"))
        return tmpfile.name


class MaskVisualizationPane(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("maskVisualizationPanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(260)
        self._html_path = _write_html_document()
        self._pending_payload: str | None = None
        self._page_ready = False

        self._web_view = None
        if QWebEngineView is not None:
            try:
                self._web_view = QWebEngineView()
                self._web_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self._web_view.loadFinished.connect(self._on_page_loaded)
                self._web_view.load(QUrl.fromLocalFile(self._html_path))
            except Exception:  # pragma: no cover - headless or webengine bootstrap failure
                self._web_view = None

        self.title_label = QLabel("3D Mask Preview")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #edf2f7;")

        self.placeholder_label = QLabel("Load a CT and masks to preview them in 3D.")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setWordWrap(True)
        self.placeholder_label.setStyleSheet(
            """
            QLabel {
                background: #0f1722;
                border: 1px solid #243244;
                border-radius: 14px;
                color: #9fb0c3;
                font-size: 13px;
                padding: 16px;
            }
            """
        )

        self.stack = QStackedWidget()
        self.stack.addWidget(self.placeholder_label)
        if self._web_view is not None:
            self._web_view.setStyleSheet(
                "background: #0f1722; border: 1px solid #243244; border-radius: 14px;"
            )
            self.stack.addWidget(self._web_view)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.stack, 1)

    def clear_view(self) -> None:
        self.placeholder_label.setText("Load a CT and masks to preview them in 3D.")
        self._pending_payload = None
        self.stack.setCurrentWidget(self.placeholder_label)

    def set_busy(self, message: str) -> None:
        self.placeholder_label.setText(message)
        self.stack.setCurrentWidget(self.placeholder_label)

    def set_data_json(self, figure_json: dict | None, visible_mask_count: int) -> None:
        if self._web_view is None:
            self.placeholder_label.setText("3D mask preview requires PyQtWebEngine.")
            self.stack.setCurrentWidget(self.placeholder_label)
            return

        if figure_json is None:
            self.clear_view()
            return

        payload = json.dumps(figure_json, separators=(",", ":"))
        self._pending_payload = payload
        if self._page_ready:
            self._push_payload()
            self.stack.setCurrentWidget(self._web_view)
        else:
            self.stack.setCurrentWidget(self.placeholder_label)

    def _on_page_loaded(self, ok: bool) -> None:
        self._page_ready = ok
        if not ok:
            self.placeholder_label.setText("3D mask preview could not be initialized.")
            self.stack.setCurrentWidget(self.placeholder_label)
            return

        self.stack.setCurrentWidget(self._web_view)
        self._push_payload()

    def _push_payload(self) -> None:
        if self._web_view is None or not self._pending_payload:
            return

        payload = self._pending_payload
        self._pending_payload = None
        self._web_view.page().runJavaScript(f"window.renderMaskPreview({json.dumps(payload)});")
