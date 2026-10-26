import io
from pathlib import Path

import numpy as np
import pytest
import trimesh

import paper3d as p3
from paper3d.export import _is_convex
from paper3d.primitives import chamfered_box
from paper3d.silhouette import extract_mask, load_image

SAMPLES = Path(__file__).parent.parent / "samples"


@pytest.fixture(scope="module")
def cat():
    return p3.generate(p3.Quadruped(), height=15)


def test_plain_box_has_six_faces():
    box = chamfered_box((0, 0, 0), (2, 3, 4))
    assert len(p3.merge_faces(box)) == 6
    assert np.allclose(box.extents, [2, 3, 4])


def test_chamfer_adds_one_face_per_edge():
    box = chamfered_box((0, 0, 0), (2, 2, 2), {"top+front": 0.5, "top+left": 0.5})
    faces = p3.merge_faces(box)
    assert len(faces) == 8
    assert box.volume == pytest.approx(8 - 2 * 2 * 0.5 * 0.5 * 0.5 + 0.5 ** 3 / 3, rel=1e-6)


def test_parts_are_closed_and_low_poly(cat):
    assert all(m.is_watertight and m.volume > 0 for m in cat.parts.values())
    assert cat.face_count < 120
    assert {"body", "head", "ear_left", "ear_right", "tail"} <= set(cat.parts)
    assert sum(n.startswith("leg_") for n in cat.parts) == 4


def test_height_and_ground(cat):
    lo, hi = cat.bounds
    assert lo[1] == pytest.approx(0, abs=1e-6)
    assert hi[1] - lo[1] == pytest.approx(15)


def test_symmetric(cat):
    lo, hi = cat.bounds
    assert lo[0] == pytest.approx(-hi[0])


def test_faces_cover_parts_and_are_convex(cat):
    for name, mesh in cat.parts.items():
        faces = cat.faces[name]
        assert sorted(t for f in faces for t in f.tris) == list(range(len(mesh.faces)))
        assert all(_is_convex(mesh.vertices[f.verts], f.normal) for f in faces)


def test_obj_roundtrip(cat):
    obj = p3.export_obj(cat.parts, cat.faces).decode()
    assert obj.count("\no ") == len(cat.parts)
    assert obj.count("\nf ") == cat.face_count
    scene = trimesh.load(io.BytesIO(obj.encode()), file_type="obj", group_material=False)
    area = sum(g.area for g in scene.geometry.values()) if isinstance(scene, trimesh.Scene) else scene.area
    assert area == pytest.approx(sum(m.area for m in cat.parts.values()), rel=1e-4)


def test_params_roundtrip():
    q = p3.Quadruped(head_w=9)
    assert p3.Quadruped.from_dict(q.to_dict() | {"junk": 1}) == q


def test_extreme_params_still_build():
    q = p3.Quadruped(**{f: getattr(p3.Quadruped(), f) for f in p3.Quadruped().to_dict()})
    q.tail_len = 0
    q.paw_len = 0
    q.head_top_side = 3
    q.head_bottom_side = 3
    m = p3.generate(q)
    assert "tail" not in m.parts and all(m.is_watertight for m in m.parts.values())


def test_reference_mask():
    rgba = load_image((SAMPLES / "cat_front34.png").read_bytes())
    mask = extract_mask(rgba)
    assert 0.2 < (mask > 0).mean() < 0.8


def test_keypoint_fit_recovers_camera_and_shape():
    from paper3d.camera import Camera, model_frame
    from paper3d.fit import fit_keypoints
    from paper3d.template import keypoints

    true_q = p3.Quadruped(body_d=12.0, head_h=6.5, leg_h=3.5)
    true_cam = Camera(az=35, el=12, dist=6, scale=0.8, cx=0.5, cy=0.5)
    kp = keypoints(true_q)
    target, height = model_frame(true_q.build())
    uv, _ = true_cam.project(np.array(list(kp.values())), target, height, 1000, 1000)
    pts = dict(zip(kp, map(tuple, uv)))

    q, cam, err = fit_keypoints(pts, 1000, 1000, p3.Quadruped(), Camera(az=25, el=5, dist=5),
                                fixed=("head_w",))
    assert err < 2
    assert q.body_d == pytest.approx(12.0, rel=0.05)
    assert q.leg_h == pytest.approx(3.5, rel=0.05)
    assert cam.az == pytest.approx(35, abs=2)


def test_overlay_draws_only_on_model():
    from paper3d.camera import Camera, overlay, silhouette
    q = p3.Quadruped()
    parts = q.build()
    faces = {n: p3.merge_faces(m) for n, m in parts.items()}
    cam = Camera()
    img = np.zeros((300, 300, 3), np.uint8)
    out = overlay(img, parts, faces, cam, color=(255, 0, 0))
    drawn = out[:, :, 0] > 0
    sil = silhouette(parts, cam, 300, 300) > 0
    import cv2
    near = cv2.dilate(sil.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    assert drawn.any() and not (drawn & ~near).any()


def test_texture_bake_and_export():
    import json
    import zipfile

    from paper3d.camera import Camera
    from paper3d.export import export_textured
    from paper3d.texture import bake

    q = p3.Quadruped()
    parts = q.build()
    faces = {n: p3.merge_faces(m) for n, m in parts.items()}
    photos = []
    for v in ("front", "back"):
        rgb = load_image((SAMPLES / f"cat_{v}34.png").read_bytes())[:, :, :3]
        photos.append((rgb, Camera(**json.loads((SAMPLES / f"cat_{v}34.camera.json").read_text()))))
    b = bake(parts, faces, photos, size=512)
    assert b.image.shape == (512, 512, 3)
    assert b.coverage > 0.4  # больше половины поверхности видно на двух фото
    for name in parts:
        assert len(b.uv[name]) == len(faces[name])
        assert all(((u >= 0) & (u <= 1)).all() for u in b.uv[name])
    files = export_textured(parts, faces, b)
    obj = files["papercraft_model.obj"].decode()
    assert "mtllib papercraft_model.mtl" in obj and "usemtl paper" in obj
    assert files["papercraft_model.png"][:4] == b"\x89PNG"
    zipfile.ZipFile(io.BytesIO(p3.to_zip(files)))
    # trimesh читает OBJ с UV и текстурой
    mesh = trimesh.load(io.BytesIO(files["papercraft_model.obj"]), file_type="obj",
                        resolver=trimesh.resolvers.ZipResolver(io.BytesIO(p3.to_zip(files))),
                        force="mesh")
    assert mesh.visual.uv is not None and len(mesh.visual.uv) == len(mesh.vertices)
