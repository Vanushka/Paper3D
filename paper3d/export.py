"""Экспорт в OBJ: копланарные треугольники сливаются в многоугольные грани,
каждая деталь модели — отдельный объект (o name)."""

import io
import zipfile
from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class Face:
    verts: list[int]        # контур грани (против часовой стрелки, если смотреть снаружи)
    normal: np.ndarray
    tris: list[int]         # треугольники сетки, из которых состоит грань


def merge_faces(mesh: trimesh.Trimesh, convex: bool = True) -> list[Face]:
    """Возвращает грани модели как многоугольники.

    Плоская область без дыр становится одной n-угольной гранью. Если у области
    есть дыры или (при convex=True) она вогнутая, её треугольники жадно
    сливаются в выпуклые куски: вогнутые n-угольники многие программы
    развёртки триангулируют неправильно.
    """
    groups = [list(f) for f in mesh.facets]
    in_facet = np.zeros(len(mesh.faces), bool)
    for g in groups:
        in_facet[g] = True
    groups += [[i] for i in np.nonzero(~in_facet)[0]]

    result = []
    for g in groups:
        normal = mesh.face_normals[g[0]]
        g = [int(i) for i in g]
        loop = _boundary_loop(mesh.faces[g])
        if loop is not None and (not convex or _is_convex(mesh.vertices[loop], normal)):
            result.append(Face(loop, normal, g))
        else:
            result += [Face(v, normal, t) for v, t in _convex_pieces(mesh, g, normal)]
    return result


def _is_convex(pts: np.ndarray, normal: np.ndarray) -> bool:
    d = np.roll(pts, -1, axis=0) - pts
    turn = np.cross(d, np.roll(d, -1, axis=0)) @ normal
    eps = 1e-9 * max(1.0, float(np.abs(pts).max())) ** 2
    return bool((turn >= -eps).all())


def _convex_pieces(mesh, tris: list[int], normal) -> list[tuple[list[int], list[int]]]:
    """Жадно сливает соседние куски, пока они остаются выпуклыми;
    сначала — по самым длинным общим рёбрам (так куски получаются крупнее)."""
    V = mesh.vertices
    pieces = [([int(v) for v in mesh.faces[t]], [t]) for t in tris]
    while True:
        best = None
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                joined = _join_loops(pieces[i][0], pieces[j][0])
                if joined is None:
                    continue
                loop, (u, w) = joined
                length = np.linalg.norm(V[u] - V[w])
                if (best is None or length > best[0]) and _is_convex(V[loop], normal):
                    best = (length, i, j, loop)
        if best is None:
            return pieces
        _, i, j, loop = best
        pieces[i] = (loop, pieces[i][1] + pieces[j][1])
        del pieces[j]


def _join_loops(a: list[int], b: list[int]):
    """Склеивает два контура по единственному общему ребру → (контур, ребро)."""
    ea = {(a[k], a[(k + 1) % len(a)]): k for k in range(len(a))}
    shared = [(m, ea[(b[(m + 1) % len(b)], b[m])]) for m in range(len(b))
              if (b[(m + 1) % len(b)], b[m]) in ea]
    if len(shared) != 1:
        return None
    mb, ka = shared[0]
    # ребро a[ka] → a[ka+1] входит в b в обратном направлении
    a_rot = a[ka + 1:] + a[:ka + 1]          # от конца ребра до его начала
    b_rot = b[mb + 1:] + b[:mb + 1]          # от a[ka] до a[ka+1]
    loop = a_rot + b_rot[1:-1]
    if len(set(loop)) != len(loop):
        return None
    return loop, (a[ka], a[(ka + 1) % len(a)])


def _boundary_loop(faces: np.ndarray) -> list[int] | None:
    edges = {(int(a), int(b)) for f in faces for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}
    boundary = [e for e in edges if (e[1], e[0]) not in edges]
    if not boundary:
        return None
    nxt = {}
    for a, b in boundary:
        if a in nxt:
            return None  # вершина-«восьмёрка»
        nxt[a] = b
    start = boundary[0][0]
    loop, cur = [start], nxt[start]
    while cur != start:
        loop.append(cur)
        cur = nxt[cur]
    if len(loop) != len(boundary):
        return None  # несколько контуров — область с дырой
    return loop


def export_obj(parts: dict[str, trimesh.Trimesh], faces: dict[str, list[Face]] | None = None,
               convex: bool = True, uv: dict[str, list[np.ndarray]] | None = None,
               mtl: str | None = None) -> bytes:
    """OBJ, где каждая деталь — отдельный объект со своими гранями-многоугольниками.

    uv — UV-координаты вершин каждой грани (из texture.bake), mtl — имя файла материалов.
    """
    faces = faces or {name: merge_faces(m, convex) for name, m in parts.items()}
    total = sum(len(f) for f in faces.values())
    lines = [f"# Paper3D: {len(parts)} деталей, {total} граней"]
    if mtl:
        lines.append(f"mtllib {mtl}")
    base, tbase = 1, 1
    for name, mesh in parts.items():
        lines.append(f"o {name}")
        lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in mesh.vertices]
        if uv is None:
            lines += ["f " + " ".join(str(base + i) for i in f.verts) for f in faces[name]]
        else:
            if mtl:
                lines.append("usemtl paper")
            for f, t in zip(faces[name], uv[name]):
                lines += [f"vt {a:.6f} {b:.6f}" for a, b in t]
                lines.append("f " + " ".join(f"{base + i}/{tbase + j}" for j, i in enumerate(f.verts)))
                tbase += len(t)
        base += len(mesh.vertices)
    return ("\n".join(lines) + "\n").encode()


def export_textured(parts, faces, bake, name: str = "papercraft_model") -> dict[str, bytes]:
    """OBJ + MTL + PNG-атлас (файлы для zip)."""
    import cv2
    ok, png = cv2.imencode(".png", bake.image[:, :, ::-1])
    mtl = (f"newmtl paper\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {name}.png\n").encode()
    obj = export_obj(parts, faces, uv=bake.uv, mtl=f"{name}.mtl")
    return {f"{name}.obj": obj, f"{name}.mtl": mtl, f"{name}.png": png.tobytes()}


def to_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in files.items():
            zf.writestr(fname, data)
    return buf.getvalue()
