"""Камера: положение, направление, view-матрица и матрица проекции."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from .projection import look_at, orthographic, perspective


def _v3(value) -> np.ndarray:
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("Ожидаются три конечные координаты")
    return v.copy()


@dataclass
class Camera:
    position: np.ndarray = field(default_factory=lambda: np.array([5., 3., 6.]))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    projection: str = "perspective"
    fov_degrees: float = 48.0
    near: float = 0.05
    far: float = 1000.0
    ortho_half_height: float = 3.0

    def __post_init__(self):
        self.position, self.target = _v3(self.position), _v3(self.target)
        if np.linalg.norm(self.target - self.position) < 1e-8:
            raise ValueError("Камера и точка наблюдения должны различаться")

    @property
    def direction(self) -> np.ndarray:
        delta = self.target - self.position
        return delta / np.linalg.norm(delta)

    def view_matrix(self) -> np.ndarray:
        # look_at автоматически выбирает вертикаль, если взгляд почти вдоль Y.
        up = np.array([0., 0., 1.]) if abs(self.direction[1]) > .985 else np.array([0., 1., 0.])
        return look_at(self.position, self.target, up)

    def projection_matrix(self, width: int, height: int) -> np.ndarray:
        aspect = max(1, width) / max(1, height)
        if self.projection == "perspective":
            return perspective(self.fov_degrees, aspect, self.near, self.far)
        if self.projection == "orthographic":
            return orthographic(self.ortho_half_height, aspect, self.near, self.far)
        raise ValueError("Проекция должна быть perspective или orthographic")

    def set_pose(self, position, direction) -> None:
        pos = _v3(position)
        dr = _v3(direction)
        length = np.linalg.norm(dr)
        if length < 1e-8:
            raise ValueError("Вектор обзора не должен быть нулевым")
        distance = max(0.2, np.linalg.norm(self.target - self.position))
        self.position = pos
        self.target = pos + dr / length * distance

    def orbit(self, yaw_degrees: float, pitch_degrees: float = 0.0) -> None:
        """Вращаем камеру вокруг неподвижной цели, не меняя модели."""
        offset = self.position - self.target
        r = np.linalg.norm(offset)
        if r < 1e-7:
            return
        yaw = math.atan2(offset[0], offset[2]) + math.radians(yaw_degrees)
        pitch = math.asin(np.clip(offset[1] / r, -1., 1.)) + math.radians(pitch_degrees)
        pitch = float(np.clip(pitch, math.radians(-88.), math.radians(88.)))
        self.position = self.target + r * np.array([
            math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)
        ])

    def zoom(self, factor: float) -> None:
        offset = self.position - self.target
        radius = max(0.15, np.linalg.norm(offset) * factor)
        radius = min(radius, 600.)
        self.position = self.target + radius * offset / np.linalg.norm(offset)
        self.ortho_half_height = float(np.clip(self.ortho_half_height * factor, .05, 300.))

    def fit(self, vertices: np.ndarray) -> None:
        positions = np.asarray(vertices, dtype=float)
        if len(positions) == 0:
            return
        self.target = (positions.max(axis=0) + positions.min(axis=0)) / 2
        radius = max(0.8, float(np.linalg.norm(positions - self.target, axis=1).max()))
        self.position = self.target + radius * np.array([2.35, 1.45, 2.85])
        self.ortho_half_height = radius * 1.65
