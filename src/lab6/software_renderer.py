"""Независимый от Qt рендер: нормали, отсечение, Z-буфер и несколько моделей."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .camera import Camera
from .scene import SceneObject
from .visibility import face_normals, is_front_facing
from .rasterizer import clip_polygon, clip_segment, to_screen, draw_triangle, draw_depth_line


@dataclass
class RenderOptions:
    cull_backfaces: bool = True
    use_zbuffer: bool = True
    draw_faces: bool = True
    draw_edges: bool = True
    draw_normals: bool = False


@dataclass
class RenderResult:
    pixels: np.ndarray
    zbuffer: np.ndarray
    total_faces: int
    culled_faces: int
    rendered_faces: int
    colored_pixels: int


def render_scene(objects: list[SceneObject], camera: Camera, width: int, height: int,
                 options: RenderOptions | None = None) -> RenderResult:
    """Один буфер глубины на все объекты: близкий фрагмент перекрывает дальний."""
    if width < 1 or height < 1:
        raise ValueError("Размер изображения должен быть положительным")
    options = options or RenderOptions()
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)
    view = camera.view_matrix()
    projection = camera.projection_matrix(width, height)
    vp = projection @ view
    total = culled = drawn_faces = colored = 0

    for obj in objects:
        mesh = obj.model
        world = mesh.world_vertices
        homogeneous = np.column_stack([world, np.ones(len(world))])
        clip_vertices = (vp @ homogeneous.T).T
        face_info = []
        normals = face_normals(mesh)
        for fi, face in enumerate(mesh.faces):
            total += 1
            normal = normals[fi]
            center = world[list(face.indices)].mean(axis=0)
            front = is_front_facing(normal, center, camera)
            if options.cull_backfaces and not front:
                culled += 1
                continue
            face_info.append((fi, face, normal, center))
            polygon = clip_polygon(clip_vertices[list(face.indices)])
            if len(polygon) < 3:
                continue
            drawn_faces += 1
            if options.draw_faces:
                screen = to_screen(polygon, width, height)
                # Умеренный постоянный цвет каждой грани (не модель освещения ЛР 9).
                hue_factor = .80 + .05 * (fi % 5)
                color = tuple(int(np.clip(channel * hue_factor + 12, 0, 255)) for channel in obj.color)
                for k in range(1, len(screen)-1):
                    triangle = screen[[0, k, k+1]]
                    colored += draw_triangle(triangle, color, pixels, zbuffer, options.use_zbuffer)

        if options.draw_edges:
            # Рёбра наложены поверх записанных ранее фрагментов с проверкой глубины.
            # Если faces выключены, всегда показывается весь каркас.
            for a, b in mesh.edges:
                seg = clip_segment(clip_vertices[a], clip_vertices[b])
                if len(seg) == 2:
                    segment = to_screen(seg, width, height)
                    draw_depth_line(segment[0], segment[1], pixels, zbuffer,
                                    (227, 237, 251),
                                    test_depth=(options.draw_faces and options.use_zbuffer))

        if options.draw_normals:
            # Нормали к граням вычисляются крестовым произведением и преобразуются (M^-1)^T.
            arrow_length = max(.16, float(np.linalg.norm(world.max(axis=0)-world.min(axis=0))) * .075)
            for fi, face, normal, center in face_info[:750]:
                if np.linalg.norm(normal) < 1e-12:
                    continue
                first = (vp @ np.append(center, 1.))
                last = (vp @ np.append(center + arrow_length * normal, 1.))
                seg = clip_segment(first, last)
                if len(seg) == 2:
                    endpoints = to_screen(seg, width, height)
                    draw_depth_line(endpoints[0], endpoints[1], pixels, zbuffer,
                                    (255, 207, 87), test_depth=False)

    return RenderResult(pixels, zbuffer, total, culled, drawn_faces, colored)
