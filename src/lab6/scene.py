"""Несколько независимых объектов: общий Z-буфер, матрица у каждого своя."""
from __future__ import annotations

from dataclasses import dataclass
from .geometry import Polyhedron, hexahedron, octahedron, tetrahedron
from .transforms import translation, scaling


@dataclass
class SceneObject:
    model: Polyhedron
    color: tuple[int, int, int] = (77, 160, 232)
    label: str = ""

    @property
    def name(self) -> str:
        return self.label or self.model.name


def overlap_scene() -> list[SceneObject]:
    """Три пересекающиеся в проекции модели разных цветов для демонстрации Z-буфера."""
    cube, tetra, octa = hexahedron(), tetrahedron(), octahedron()
    cube.apply(translation(0.20, 0.00, 0.50) @ scaling(.95, .95, .95))
    tetra.apply(translation(0.60, 0.10, -0.80) @ scaling(.9, .9, .9))
    octa.apply(translation(-0.40, -0.10, -1.00) @ scaling(.9, .9, .9))
    return [
        SceneObject(cube, (69, 149, 239), "Синий куб"),
        SceneObject(tetra, (239, 118, 115), "Красный тетраэдр"),
        SceneObject(octa, (82, 206, 167), "Зелёный октаэдр"),
    ]
