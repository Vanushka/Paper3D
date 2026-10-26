"""Перенос цветов с фото на грани модели: текстура-атлас + UV-координаты.

Каждая грань получает свой прямоугольник в атласе. Пиксель атласа — точка на грани;
её цвет берётся с того фото, где грань видна прямее всего (z-буфер исключает
перекрытые места). Грани, не видные ни на одном фото, берут цвет с зеркальной
грани (модель симметрична по X), остальные пустоты заполняются цветом соседних пикселей.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Camera, face_buffer, model_frame

PAD = 3            # поля вокруг грани в атласе, пикселей (против швов при фильтрации)
EDGE_GUARD = 0.004  # у краёв граней на фото не берём цвет (доля высоты фото): там ошибка подгонки


@dataclass
class Bake:
    image: np.ndarray                    # атлас RGB uint8
    uv: dict[str, list[np.ndarray]]      # деталь → по грани: UV вершин контура (k×2, 0..1, v вверх)
    coverage: float                      # доля пикселей граней, взятых прямо с фото


def bake(parts, faces, photos: list[tuple[np.ndarray, Camera]], size: int = 4096) -> Bake:
    """parts/faces — модель в тех же единицах, что и при подгонке камер (до масштабирования)."""
    target, height = model_frame(parts)
    names = list(parts)
    ids, n = {}, 0
    for name in names:
        for k in range(len(faces[name])):
            ids[name, k] = n
            n += 1

    # для каждого фото: номер видимой грани в каждом пикселе (у краёв граней — −2)
    views = []
    for rgb, cam in photos:
        h, w = rgb.shape[:2]
        idmap, _ = face_buffer(parts, faces, cam, w, h)
        r = max(1, round(EDGE_GUARD * h))
        kernel = np.ones((2 * r + 1,) * 2, np.uint8)
        f32 = idmap.astype(np.float32)
        edge = cv2.dilate(f32, kernel) != cv2.erode(f32, kernel)
        guarded = np.where(edge, -2, idmap)
        rgb = _remove_fold_lines(rgb, edge, r)
        # фон фото (там модель «вылезает» за кота из-за ошибки подгонки) — не брать
        bg = _background(rgb)
        idmap = np.where(bg, -2, idmap)
        guarded = np.where(bg, -2, guarded)
        eye = target + cam.dist * height * cam.rotation()[2]  # положение камеры
        views.append((rgb, cam, idmap, guarded, eye))

    # раскладка граней в атласе
    tiles = []
    for name in names:
        for k, f in enumerate(faces[name]):
            P = parts[name].vertices[f.verts]
            u = P[1] - P[0]
            u /= np.linalg.norm(u)
            v = np.cross(f.normal, u)
            q = (P - P[0]) @ np.stack([u, v]).T
            lo = q.min(0)
            tiles.append(dict(name=name, k=k, face=f, P=P, origin=P[0] + lo[0] * u + lo[1] * v,
                              u=u, v=v, q=q - lo, ext=q.max(0) - lo))
    ppu, pos = _pack([t["ext"] for t in tiles], size)

    atlas = np.zeros((size, size, 3), np.uint8)
    part_px: dict[str, list] = {}
    empty = []  # грани, не видные ни на одном фото
    uv: dict[str, list] = {name: [None] * len(faces[name]) for name in names}
    direct = total = 0
    for t, (x0, y0) in zip(tiles, pos):
        tw, th = np.ceil(t["ext"] * ppu).astype(int) + 2 * PAD
        # центры пикселей тайла → точки на плоскости грани
        gx, gy = np.meshgrid(np.arange(tw) + 0.5 - PAD, np.arange(th) + 0.5 - PAD)
        pts = t["origin"] + (gx.ravel()[:, None] * t["u"] + gy.ravel()[:, None] * t["v"]) / ppu
        poly = np.zeros((th, tw), np.uint8)
        cv2.fillPoly(poly, [np.round((t["q"] * ppu + PAD) * 4).astype(np.int32)], 1, shift=2)
        col, got = _sample(t, pts, views, ids, names, parts, faces, target, height)
        col, got = col.reshape(th, tw, 3), got.reshape(th, tw)
        if not got.any():  # совсем не видна — берём и пиксели у краёв граней
            col, got = _sample(t, pts, views, ids, names, parts, faces, target, height, guard=False)
            col, got = col.reshape(th, tw, 3), got.reshape(th, tw)
        sl = np.s_[y0:y0 + th, x0:x0 + tw]
        direct += int((got & (poly > 0)).sum())
        total += int(poly.sum())
        if got.any():
            col = _push_pull(col, got)
            part_px.setdefault(t["name"], []).append(col[got & (poly > 0)])
        else:
            empty.append((t, sl))
        atlas[sl] = col
        uv[t["name"]][t["k"]] = np.column_stack([(x0 + PAD + t["q"][:, 0] * ppu) / size,
                                                 1 - (y0 + PAD + t["q"][:, 1] * ppu) / size])

    allpx = np.vstack([c for cs in part_px.values() for c in cs]) if part_px else np.full((1, 3), 128)
    for t, sl in empty:
        px = np.vstack(part_px[t["name"]]) if t["name"] in part_px else allpx
        atlas[sl] = np.median(px, axis=0).astype(np.uint8)
    return Bake(atlas, uv, direct / max(total, 1))


def _sample(t, pts, views, ids, names, parts, faces, target, height, guard=True):
    """Цвет точек грани: с лучших ракурсов, затем с зеркальной грани."""
    col = np.zeros((len(pts), 3), np.uint8)
    got = np.zeros(len(pts), bool)
    cands = []
    mirror = _mirror_face(t, parts, faces)
    for vi, (rgb, cam, idmap, guarded, eye) in enumerate(views):
        c = t["P"].mean(0)
        cands.append((t["face"].normal @ _unit(eye - c), vi, False))
        if mirror is not None:
            mc = c * [-1, 1, 1]
            mn = t["face"].normal * [-1, 1, 1]
            cands.append((mn @ _unit(eye - mc) - 0.3, vi, True))  # зеркало — только если прямого нет
    for score, vi, mirrored in sorted(cands, reverse=True):
        if score < 0.1:
            break
        rgb, cam, idmap, guarded, _ = views[vi]
        buf = guarded if guard else idmap
        h, w = rgb.shape[:2]
        p = pts * [-1, 1, 1] if mirrored else pts
        want = ids[mirror] if mirrored else ids[t["name"], t["k"]]
        px, _ = cam.project(p, target, height, w, h)
        xi, yi = np.floor(px).astype(int).T
        ok = ~got & (xi >= 0) & (yi >= 0) & (xi < w) & (yi < h)
        ok[ok] = buf[yi[ok], xi[ok]] == want
        if ok.any():
            col[ok] = _bilinear(rgb, px[ok])
            got |= ok
        if got.all():
            break
    return col, got


def _remove_fold_lines(rgb: np.ndarray, edge: np.ndarray, r: int) -> np.ndarray:
    """Стирает нарисованные на фото линии сгиба (тонкие светлые штрихи) возле рёбер модели.

    Ищем только у рёбер: так усы и прочие светлые штрихи в середине граней остаются.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thin = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((2 * r + 3,) * 2, np.uint8)) > 40
    near = cv2.dilate(edge.astype(np.uint8), np.ones((8 * r + 1,) * 2, np.uint8)) > 0
    mask = cv2.dilate((thin & near).astype(np.uint8), np.ones((3, 3), np.uint8))
    if not mask.any():
        return rgb
    return cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA)


def _background(rgb: np.ndarray) -> np.ndarray:
    """Светлый фон, связанный с краями кадра."""
    light = (rgb.min(2) > 225) & (rgb.max(2).astype(int) - rgb.min(2) < 20)
    n, labels = cv2.connectedComponents(light.astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    border = border[border > 0]
    return np.isin(labels, border)


def _mirror_face(t, parts, faces):
    """(деталь, номер) грани, симметричной данной относительно плоскости X = 0."""
    name = t["name"]
    other = name.replace("left", "#").replace("right", "left").replace("#", "right")
    if other not in parts:
        return None
    c = t["P"].mean(0) * [-1, 1, 1]
    nrm = t["face"].normal * [-1, 1, 1]
    best = None
    for k, f in enumerate(faces[other]):
        d = np.linalg.norm(parts[other].vertices[f.verts].mean(0) - c)
        if f.normal @ nrm > 0.99 and (best is None or d < best[0]):
            best = (d, k)
    return (other, best[1]) if best and best[0] < 1e-3 * max(1.0, np.abs(c).max()) else None


def _unit(v):
    return v / np.linalg.norm(v)


def _bilinear(img: np.ndarray, px: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(px[:, 0] - 0.5, 0, w - 1.001)
    y = np.clip(px[:, 1] - 0.5, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    img = img.astype(np.float32)
    c = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
         + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return np.clip(c + 0.5, 0, 255).astype(np.uint8)


def _pack(exts: list[np.ndarray], size: int):
    """Полочная укладка прямоугольников граней; подбирает масштаб (пикселей на единицу)."""
    def place(ppu):
        order = sorted(range(len(exts)), key=lambda i: -exts[i][1])
        pos = [None] * len(exts)
        x = y = shelf = 0
        for i in order:
            tw, th = np.ceil(exts[i] * ppu).astype(int) + 2 * PAD
            if x + tw > size:
                x, y, shelf = 0, y + shelf, 0
            if y + th > size or tw > size:
                return None
            pos[i] = (x, y)
            x += tw
            shelf = max(shelf, th)
        return pos

    lo, hi = 1.0, size / max(max(e) for e in exts)
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if place(mid) else (lo, mid)
    return lo, place(lo)


def _push_pull(img: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Плавно заполняет пустые пиксели усреднёнными цветами известных (пирамида «push-pull»)."""
    if known.all():
        return img
    w = known.astype(np.float32)
    c = img.astype(np.float32) * w[..., None]
    levels = []
    while True:
        levels.append((c, w))
        h, wd = w.shape
        if h <= 1 and wd <= 1:
            break
        nh, nw = max(1, (h + 1) // 2), max(1, (wd + 1) // 2)
        c = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)
        w = cv2.resize(w, (nw, nh), interpolation=cv2.INTER_AREA)
    c, w = levels[-1]
    filled = c / np.maximum(w, 1e-6)[..., None]
    for c, w in reversed(levels[:-1]):
        up = cv2.resize(filled, (w.shape[1], w.shape[0]), interpolation=cv2.INTER_LINEAR)
        a = np.clip(w, 0, 1)[..., None]
        filled = c / np.maximum(w, 1e-6)[..., None] * a + up * (1 - a)
    out = img.copy()
    out[~known] = np.clip(filled[~known] + 0.5, 0, 255).astype(np.uint8)
    return out
