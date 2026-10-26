"""3D-просмотр на three.js: модель с настоящей текстурой (Plotly текстуры не умеет)."""

import base64
import json

import cv2
import numpy as np

THREE = "https://cdn.jsdelivr.net/npm/three@0.160.0"


def viewer_html(parts, faces, bake=None, edges: bool = True, height: int = 600,
                view=(33.0, 13.0), color: str = "#8c9096", background: str = "#eef0f3") -> str:
    """HTML-страница с моделью. view — (поворот, наклон) камеры в градусах, как у фото."""
    pos, uv, nrm, lines = [], [], [], []
    for name, mesh in parts.items():
        V = mesh.vertices
        for k, f in enumerate(faces[name]):
            P = V[f.verts]
            T = bake.uv[name][k] if bake is not None else np.zeros((len(P), 2))
            for i in range(1, len(P) - 1):  # грани выпуклые — веер треугольников
                for j in (0, i, i + 1):
                    pos += list(P[j])
                    uv += list(T[j])
                    nrm += list(f.normal)
            for i in range(len(P)):
                lines += list(P[i]) + list(P[(i + 1) % len(P)])
    lo = np.min([m.bounds[0] for m in parts.values()], axis=0)
    hi = np.max([m.bounds[1] for m in parts.values()], axis=0)
    tex = ""
    if bake is not None:
        img = bake.image
        if img.shape[0] > 2048:  # для просмотра хватит 2048, а страница станет легче
            img = cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        tex = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    data = dict(pos=np.round(pos, 5).tolist(), uv=np.round(uv, 6).tolist(), nrm=np.round(nrm, 4).tolist(),
                lines=np.round(lines, 5).tolist(), center=((lo + hi) / 2).tolist(), size=float(np.max(hi - lo)),
                view=list(view), tex=tex, edges=edges, color=color)
    return _PAGE.replace("__DATA__", json.dumps(data)).replace("__THREE__", THREE) \
        .replace("__H__", str(height)).replace("__BG__", background)


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<style>html,body{margin:0;height:__H__px;background:__BG__;overflow:hidden}canvas{display:block}
#msg{position:absolute;top:45%;width:100%;text-align:center;font:14px sans-serif;color:#666}</style>
<script type="importmap">{"imports":{"three":"__THREE__/build/three.module.js",
"three/addons/":"__THREE__/examples/jsm/"}}</script>
</head><body><div id="msg">Загрузка 3D…</div>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const D = __DATA__;
document.getElementById("msg").remove();
const W = () => document.body.clientWidth, H = __H__;
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(W(), H);
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color("__BG__");
const camera = new THREE.PerspectiveCamera(30, W() / H, D.size / 100, D.size * 50);
const c = new THREE.Vector3(...D.center);
const az = D.view[0] * Math.PI / 180, el = D.view[1] * Math.PI / 180, r = D.size * 2.6;
camera.position.set(c.x + r * Math.sin(az) * Math.cos(el), c.y + r * Math.sin(el), c.z + r * Math.cos(az) * Math.cos(el));
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(c);
controls.enableDamping = true;
controls.update();

const g = new THREE.BufferGeometry();
g.setAttribute("position", new THREE.Float32BufferAttribute(D.pos, 3));
g.setAttribute("normal", new THREE.Float32BufferAttribute(D.nrm, 3));
g.setAttribute("uv", new THREE.Float32BufferAttribute(D.uv, 2));
let mat;
if (D.tex) {
  const t = new THREE.TextureLoader().load(D.tex, () => renderer.render(scene, camera));
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  t.flipY = true;
  // на фото свет уже «нарисован» — материал без освещения, цвета как на фото
  mat = new THREE.MeshBasicMaterial({map: t});
} else {
  mat = new THREE.MeshLambertMaterial({color: D.color});
  scene.add(new THREE.AmbientLight(0xffffff, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 2.0);
  sun.position.set(0.4, 0.8, 0.5);
  scene.add(sun);
}
mat.polygonOffset = true; mat.polygonOffsetFactor = 1; mat.polygonOffsetUnits = 1;
scene.add(new THREE.Mesh(g, mat));
if (D.edges) {
  const lg = new THREE.BufferGeometry();
  lg.setAttribute("position", new THREE.Float32BufferAttribute(D.lines, 3));
  scene.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({color: 0xdd2222})));
}
window.addEventListener("resize", () => {
  camera.aspect = W() / H; camera.updateProjectionMatrix(); renderer.setSize(W(), H);
});
(function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
</script></body></html>
"""
