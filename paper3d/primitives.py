"""Простые выпуклые детали: параллелепипед со скосами рёбер, пирамида.

Оси: X — вправо, Y — вверх, Z — вперёд.
"""

import numpy as np
import trimesh
from scipy.spatial import HalfspaceIntersection

# грани параллелепипеда: имя → (нормаль)
SIDES = {
    "right": (1, 0, 0), "left": (-1, 0, 0),
    "top": (0, 1, 0), "bottom": (0, -1, 0),
    "front": (0, 0, 1), "back": (0, 0, -1),
}


def convex_hull(points) -> trimesh.Trimesh:
    pts = np.asarray(points, float)
    # почти совпадающие точки дают вырожденные грани — склеиваем их
    tol = 1e-6 * max(1.0, float(np.abs(pts).max()))
    pts = np.unique(np.round(pts / tol) * tol, axis=0)
    pts = _extreme_points(pts)
    mesh = trimesh.convex.convex_hull(pts)
    mesh.merge_vertices()
    return mesh


def _extreme_points(pts: np.ndarray) -> np.ndarray:
    """Убирает точки, лежащие на рёбрах и гранях оболочки: иначе грань дробится на куски."""
    from scipy.spatial import ConvexHull
    vol = ConvexHull(pts).volume
    keep = np.ones(len(pts), bool)
    for i in range(len(pts)):
        keep[i] = False
        if keep.sum() < 4 or ConvexHull(pts[keep]).volume < vol * (1 - 1e-6):
            keep[i] = True
    return pts[keep]


def chamfered_box(lo, hi, chamfers: dict[str, float] | None = None) -> trimesh.Trimesh:
    """Параллелепипед [lo, hi] со скошенными рёбрами.

    chamfers: {"top+front": 0.8, "top+left": 1.0, ...} — ребро между двумя гранями
    и сколько срезать вдоль каждой из них.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    normals, offsets = [], []
    for n in SIDES.values():
        n = np.array(n, float)
        normals.append(n)
        offsets.append(n @ np.where(n > 0, hi, lo))  # n·p ≤ offset

    for key, c in (chamfers or {}).items():
        a, b = key.split("+")
        na, nb = np.array(SIDES[a], float), np.array(SIDES[b], float)
        oa, ob = na @ np.where(na > 0, hi, lo), nb @ np.where(nb > 0, hi, lo)
        size = min(abs(hi - lo)[np.abs(na) > 0][0], abs(hi - lo)[np.abs(nb) > 0][0])
        if c < 0.01 * size:  # слишком мелкий скос — только лишние узкие грани
            continue
        c = min(c, 0.5 * size)  # 0.5 — два скоса сходятся в конёк
        n = na + nb
        normals.append(n / np.linalg.norm(n))
        offsets.append((oa + ob - c) / np.linalg.norm(n))

    normals, offsets = np.array(normals), np.array(offsets)
    halfspaces = np.column_stack([normals, -offsets])  # A·p + b ≤ 0
    inner = _interior_point(normals, offsets, (lo + hi) / 2)
    pts = HalfspaceIntersection(halfspaces, inner).intersections
    return convex_hull(pts)


def _interior_point(normals, offsets, guess):
    """Точка строго внутри многогранника (центр Чебышёва)."""
    from scipy.optimize import linprog
    norm = np.linalg.norm(normals, axis=1)
    res = linprog(c=[0, 0, 0, -1], A_ub=np.column_stack([normals, norm]), b_ub=offsets,
                  bounds=[(None, None)] * 3 + [(0, None)])
    return res.x[:3] if res.success and res.x[3] > 1e-9 else guess


def pyramid(base_lo, base_hi, apex) -> trimesh.Trimesh:
    """Пирамида с прямоугольным основанием в плоскости y = base_lo[1]."""
    (x0, y, z0), (x1, _, z1) = base_lo, base_hi
    pts = [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1), apex]
    return convex_hull(pts)


def union(*meshes: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = trimesh.boolean.union(list(meshes), engine="manifold")
    mesh.merge_vertices()
    return mesh


def clip(mesh: trimesh.Trimesh, normal, origin) -> trimesh.Trimesh:
    """Отрезает от выпуклой детали всё, что лежит по нормали от плоскости (normal·(p − origin) > 0)."""
    n, o = np.asarray(normal, float), np.asarray(origin, float)
    d = (mesh.vertices - o) @ n
    keep = list(mesh.vertices[d <= 1e-9])
    for a, b in mesh.edges_unique:
        if (d[a] < 0) != (d[b] < 0) and abs(d[a] - d[b]) > 1e-12:
            t = d[a] / (d[a] - d[b])
            keep.append(mesh.vertices[a] + t * (mesh.vertices[b] - mesh.vertices[a]))
    return convex_hull(keep)


def rotate_x(mesh: trimesh.Trimesh, deg: float, pivot) -> trimesh.Trimesh:
    tf = trimesh.transformations.rotation_matrix(np.radians(deg), [1, 0, 0], pivot)
    return mesh.copy().apply_transform(tf)
