"""Перспективная камера: проекция модели на фото, силуэт и каркас поверх изображения."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import trimesh


@dataclass
class Camera:
    az: float = 30.0       # поворот вокруг вертикали, ° (0 — строго спереди, + — сдвиг к правому боку кадра)
    el: float = 12.0       # взгляд сверху, °
    roll: float = 0.0      # наклон кадра, °
    dist: float = 4.0      # удалённость в высотах модели (больше — меньше перспективы)
    scale: float = 0.8     # высота модели в долях высоты кадра
    cx: float = 0.5        # где в кадре центр модели (доли ширины/высоты)
    cy: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    def rotation(self) -> np.ndarray:
        az, el, roll = np.radians([self.az, self.el, self.roll])
        d = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])  # к камере
        right = np.cross([0.0, 1.0, 0.0], d)
        right /= np.linalg.norm(right)
        up = np.cross(d, right)
        c, s = np.cos(roll), np.sin(roll)
        return np.stack([c * right + s * up, -s * right + c * up, d])

    def project(self, pts: np.ndarray, target: np.ndarray, height: float, w: int, h: int):
        """Точки модели → (пиксели N×2, глубина N; больше — ближе к камере)."""
        pc = (pts - target) @ self.rotation().T
        D = self.dist * height
        depth = D - pc[:, 2]
        f = self.scale * h * D / height
        uv = np.column_stack([self.cx * w + f * pc[:, 0] / depth, self.cy * h - f * pc[:, 1] / depth])
        return uv, -depth


def model_frame(parts: dict[str, trimesh.Trimesh]):
    """Опорная точка (центр габаритов) и высота модели."""
    lo = np.min([m.bounds[0] for m in parts.values()], axis=0)
    hi = np.max([m.bounds[1] for m in parts.values()], axis=0)
    return (lo + hi) / 2, hi[1] - lo[1]


def silhouette(parts, cam: Camera, w: int, h: int) -> np.ndarray:
    """Маска модели в кадре (uint8 0/255)."""
    target, height = model_frame(parts)
    mask = np.zeros((h, w), np.uint8)
    for mesh in parts.values():
        uv, _ = cam.project(mesh.vertices, target, height, w, h)
        tri = np.round(uv[mesh.faces] * 4).astype(np.int32)  # субпиксельная точность
        cv2.fillPoly(mask, list(tri), 255, lineType=cv2.LINE_8, shift=2)
    return mask


def face_buffer(parts, faces, cam: Camera, w: int, h: int):
    """Z-буфер по граням: для каждого пикселя — номер видимой грани (−1 — фон).

    Возвращает (idmap, список граней [(деталь, Face, uv-контур)]).
    Глубина интерполируется как 1/z — для перспективы это точно.
    """
    target, height = model_frame(parts)
    idmap = np.full((h, w), -1, np.int32)
    zbuf = np.full((h, w), -np.inf)
    items = []
    for name, mesh in parts.items():
        uv, z = cam.project(mesh.vertices, target, height, w, h)
        inv = 1.0 / np.maximum(-z, 1e-9)  # 1/глубина: больше — ближе
        for f in faces[name]:
            i = len(items)
            items.append((name, f, uv[f.verts]))
            pts = uv[f.verts]
            x0, y0 = np.floor(pts.min(0)).astype(int)
            x1, y1 = np.ceil(pts.max(0)).astype(int) + 1
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if x0 >= x1 or y0 >= y1:
                continue
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [np.round((pts - [x0, y0]) * 4).astype(np.int32)], 1, shift=2)
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            # плоскость 1/z = a·u + b·v + c по трём вершинам грани
            A = np.column_stack([pts[:3], np.ones(3)])
            try:
                coef = np.linalg.solve(A, inv[f.verts][:3])
            except np.linalg.LinAlgError:
                continue
            xs, ys = xs + x0, ys + y0
            d = coef[0] * xs + coef[1] * ys + coef[2]
            win = d > zbuf[ys, xs]
            zbuf[ys[win], xs[win]] = d[win]
            idmap[ys[win], xs[win]] = i
    return idmap, items


def overlay(image: np.ndarray, parts, faces, cam: Camera, color=(255, 40, 40)) -> np.ndarray:
    """Видимые рёбра граней модели поверх фото."""
    h, w = image.shape[:2]
    idmap, items = face_buffer(parts, faces, cam, w, h)
    thick = max(1, round(min(w, h) / 400))
    kernel = np.ones((2 * thick + 3,) * 2, np.uint8)
    edges = np.zeros((h, w), np.uint8)
    for i in np.unique(idmap):
        if i < 0:
            continue
        own = cv2.dilate((idmap == i).astype(np.uint8), kernel)  # ребро видно там, где видна грань
        layer = np.zeros((h, w), np.uint8)
        cv2.polylines(layer, [np.round(items[i][2] * 4).astype(np.int32)], True, 1, thick, cv2.LINE_8, shift=2)
        edges |= layer & own
    out = image.copy()
    out[edges > 0] = color
    return out
