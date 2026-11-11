"""Попиксельная растеризация треугольников и алгоритм Z-буфера (без OpenGL)."""
from __future__ import annotations

import numpy as np

CLIP_PLANES = np.array([
    [1., 0., 0., 1.], [-1., 0., 0., 1.],
    [0., 1., 0., 1.], [0., -1., 0., 1.],
    [0., 0., 1., 1.], [0., 0., -1., 1.],
])


def clip_polygon(vertices: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman по всем 6 границам однородного пространства."""
    polygon = np.asarray(vertices, float).reshape(-1, 4)
    for plane in CLIP_PLANES:
        if not len(polygon):
            break
        output = []
        prev = polygon[-1]
        prev_dot = float(prev @ plane)
        for curr in polygon:
            curr_dot = float(curr @ plane)
            if (prev_dot >= 0) != (curr_dot >= 0):
                t = prev_dot / (prev_dot - curr_dot)
                output.append(prev + t * (curr - prev))
            if curr_dot >= 0:
                output.append(curr)
            prev, prev_dot = curr, curr_dot
        polygon = np.asarray(output, float).reshape(-1, 4)
    return polygon


def clip_segment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Отсечение отрезка в clip-space с сохранением его концов."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    begin, end = 0., 1.
    delta = b - a
    for plane in CLIP_PLANES:
        initial, slope = float(a @ plane), float(delta @ plane)
        if abs(slope) < 1e-12:
            if initial < 0:
                return np.empty((0, 4))
        else:
            t = -initial / slope
            if slope > 0:
                begin = max(begin, t)
            else:
                end = min(end, t)
            if begin > end:
                return np.empty((0, 4))
    return np.array([a + begin * delta, a + end * delta])


def to_screen(clip: np.ndarray, width: int, height: int) -> np.ndarray:
    """Переход в пиксели; z = NDC глубина (меньшее значение ближе)."""
    ndc = clip[:, :3] / clip[:, 3, None]
    return np.column_stack(((ndc[:, 0] + 1) * width * .5,
                            (1 - ndc[:, 1]) * height * .5, ndc[:, 2]))


def draw_triangle(p: np.ndarray, rgb: tuple[int, int, int],
                  pixels: np.ndarray, depth_buffer: np.ndarray,
                  use_zbuffer: bool = True) -> int:
    """Барицентрические веса -> z(x,y) -> сравнение с z-буфером -> запись цвета."""
    height, width = depth_buffer.shape
    xmin = max(0, int(np.floor(p[:, 0].min())))
    xmax = min(width - 1, int(np.ceil(p[:, 0].max())))
    ymin = max(0, int(np.floor(p[:, 1].min())))
    ymax = min(height - 1, int(np.ceil(p[:, 1].max())))
    if xmin > xmax or ymin > ymax:
        return 0
    (x0, y0), (x1, y1), (x2, y2) = p[:, :2]
    area = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)
    if abs(area) < 1e-9:
        return 0
    yy, xx = np.ogrid[ymin:ymax+1, xmin:xmax+1]
    xx, yy = xx + .5, yy + .5
    w0 = ((x1-xx)*(y2-yy) - (y1-yy)*(x2-xx)) / area
    w1 = ((x2-xx)*(y0-yy) - (y2-yy)*(x0-xx)) / area
    w2 = 1. - w0 - w1
    inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
    if not np.any(inside):
        return 0
    # Для z в NDC глубина линейно интерполируется в экранном пространстве.
    z = w0*p[0, 2] + w1*p[1, 2] + w2*p[2, 2]
    depth = depth_buffer[ymin:ymax+1, xmin:xmax+1]
    drawn = inside & (z < depth) if use_zbuffer else inside
    if not np.any(drawn):
        return 0
    depth[drawn] = z[drawn]
    rgba = pixels[ymin:ymax+1, xmin:xmax+1]
    rgba[drawn] = (*rgb, 255)
    return int(np.count_nonzero(drawn))


def draw_depth_line(p0: np.ndarray, p1: np.ndarray,
                    pixels: np.ndarray, zbuffer: np.ndarray,
                    color: tuple[int, int, int] = (204, 225, 250),
                    test_depth: bool = True) -> None:
    height, width = zbuffer.shape
    length = int(max(abs(p1[0]-p0[0]), abs(p1[1]-p0[1]))) + 1
    if length > 8000:
        return
    t = np.linspace(0., 1., max(2, length))
    points = p0[None, :] * (1.-t[:, None]) + p1[None, :] * t[:, None]
    x = np.rint(points[:, 0]).astype(int)
    y = np.rint(points[:, 1]).astype(int)
    allowed = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, z = x[allowed], y[allowed], points[allowed, 2]
    if test_depth:
        visible = z <= zbuffer[y, x] + 0.00008
        x, y = x[visible], y[visible]
    pixels[y, x] = (*color, 255)
