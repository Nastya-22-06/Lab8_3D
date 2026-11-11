"""Внешние нормали и back-face culling для обоих видов проекции."""
from __future__ import annotations

import numpy as np
from .camera import Camera
from .geometry import Polyhedron


def face_world_normal(model: Polyhedron, face_index: int) -> np.ndarray:
    """Нормаль через (M^-1)^T: работает и при зеркальных отражениях."""
    local = np.asarray(model.faces[face_index].outward_normal if model.faces[face_index].outward_normal is not None
                       else model.faces[face_index].normal(model.local_vertices))
    linear = model.model_matrix[:3, :3]
    # Псевдообратная матрица допустима даже после нулевого масштабирования.
    transformed = np.linalg.pinv(linear).T @ local
    norm = float(np.linalg.norm(transformed))
    return transformed / norm if norm > 1e-12 else np.zeros(3)


def is_front_facing(normal: np.ndarray, center: np.ndarray, camera: Camera) -> bool:
    if camera.projection == "orthographic":
        to_eye = -camera.direction
    else:
        to_eye = camera.position - center
    return float(np.dot(normal, to_eye)) > 1e-10


def face_normals(model: Polyhedron) -> np.ndarray:
    """Преобразуем все нормали за одну операцию, не вычисляя inverse для каждой грани."""
    local = np.array([f.outward_normal if f.outward_normal is not None
                      else f.normal(model.local_vertices) for f in model.faces], dtype=float)
    mat = np.linalg.pinv(model.model_matrix[:3, :3]).T
    result = (mat @ local.T).T
    lengths = np.linalg.norm(result, axis=1)
    mask = lengths > 1e-12
    result[mask] /= lengths[mask, None]
    return result

