"""Виджет отображения: Qt показывает результат нашего программного растеризатора."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .camera import Camera
from .geometry import Polyhedron
from .projection import project
from .scene import SceneObject
from .software_renderer import RenderOptions, render_scene


class Viewport(QWidget):
    camera_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = Camera()
        self.items: list[SceneObject] = []
        self.polyhedron: Polyhedron | None = None  # Совместимость с ЛР 6–7
        self.show_faces = True
        self.show_edges = True
        self.show_axes = True
        self.cull_backfaces = True
        self.use_zbuffer = True
        self.show_normals = False
        self._mouse_pos = None
        self.setMinimumSize(500, 450)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setMouseTracking(True)

    @property
    def mode(self) -> str:
        return self.camera.projection

    @mode.setter
    def mode(self, projection_type: str) -> None:
        self.camera.projection = projection_type
        self.update()

    def set_polyhedron(self, model: Polyhedron) -> None:
        self.set_scene([SceneObject(model)])

    def set_scene(self, items: list[SceneObject], fit: bool = True) -> None:
        self.items = list(items)
        self.polyhedron = self.items[0].model if self.items else None
        if fit:
            self.fit_camera()
        else:
            self.update()

    def fit_camera(self) -> None:
        if self.items:
            points = np.concatenate([item.model.world_vertices for item in self.items])
            self.camera.fit(points)
        self.update()
        self.camera_changed.emit()

    def reset_camera(self) -> None:
        self.fit_camera()

    def _draw_background(self, painter: QPainter):
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0., QColor("#10213b"))
        gradient.setColorAt(1., QColor("#09111e"))
        painter.fillRect(self.rect(), QBrush(gradient))
        if not self.show_axes:
            return
        view = self.camera.view_matrix()
        projection = self.camera.projection_matrix(self.width(), self.height())
        center = self.camera.target
        extent = max(3., self.camera.ortho_half_height * 1.6)

        def line(a, b, color, pen_width=1):
            coords, valid = project(np.array([a, b], float), view, projection, self.width(), self.height())
            if np.all(valid) and np.isfinite(coords).all():
                pen = QPen(QColor(color))
                pen.setWidthF(pen_width)
                painter.setPen(pen)
                painter.drawLine(QPointF(*coords[0]), QPointF(*coords[1]))
                return QPointF(*coords[1])
            return None

        for k in range(-7, 8):
            x = k * extent / 7
            line([x, -1.9, -extent], [x, -1.9, extent], "#1c3453", .8)
            line([-extent, -1.9, x], [extent, -1.9, x], "#1c3453", .8)
        for tip, color, label in [([2.2, 0, 0], "#ff8497", "X"),
                                  ([0, 2.2, 0], "#6ee5bd", "Y"),
                                  ([0, 0, 2.2], "#79b8ff", "Z")]:
            end = line([0., 0., 0.], tip, color, 1.8)
            if end is not None:
                painter.drawText(end + QPointF(5., -5.), label)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_background(painter)
        ratio = min(1., 740. / max(1, self.width()), 520. / max(1, self.height()))
        w, h = max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio))
        opts = RenderOptions(self.cull_backfaces, self.use_zbuffer,
                             self.show_faces, self.show_edges, self.show_normals)
        result = render_scene(self.items, self.camera, w, h, opts)
        image = QImage(result.pixels.data, w, h, result.pixels.strides[0],
                       QImage.Format.Format_RGBA8888).copy()
        painter.drawImage(self.rect(), image)
        painter.setFont(QFont("Arial", 10))
        painter.setPen(QColor("#e9f2ff"))
        projection = "ПЕРСПЕКТИВА" if self.camera.projection == "perspective" else "ПАРАЛЛЕЛЬНАЯ"
        painter.drawText(16, 26, f"{projection}  ·  ЛР 8 / программный Z-буфер")
        painter.setPen(QColor("#a6b8d2"))
        painter.drawText(16, 47, f"Грани: {result.total_faces}  |  нелицевые: {result.culled_faces}  |  видимые: {result.rendered_faces}")
        painter.drawText(16, 66, f"Back-face culling: {'ON' if self.cull_backfaces else 'OFF'}  ·  Z-buffer: {'ON' if self.use_zbuffer else 'OFF'}")
        painter.drawText(16, self.height()-16, "ЛКМ: вращать камеру  ·  Колесо: приблизить/отдалить")
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._mouse_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._mouse_pos
            self.camera.orbit(delta.x() * .44, -delta.y() * .44)
            self._mouse_pos = event.position()
            self.update()
            self.camera_changed.emit()

    def mouseReleaseEvent(self, event):
        self._mouse_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event):
        self.camera.zoom(.87 if event.angleDelta().y() > 0 else 1.15)
        self.update()
        self.camera_changed.emit()
