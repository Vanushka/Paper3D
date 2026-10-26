import json
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import paper3d as p3
from paper3d.camera import Camera, overlay
from paper3d.silhouette import load_image
from paper3d.export import export_textured
from paper3d.texture import bake
from paper3d.viewer import viewer_html

SAMPLES = Path(__file__).parent / "samples"

st.set_page_config(
    page_title="Paper3D AI",
    page_icon="📐",
    layout="wide"
)

st.title("📐 Paper3D AI")
st.caption("Low-poly модели для Papercraft из простых деталей")

# центральная колонка с 3D-просмотром едет за прокруткой (боковые колонки длинные)
st.markdown("""
<style>
div[data-testid="stColumn"]:has(.st-key-viewer) {
    position: sticky;
    top: 4rem;
    align-self: flex-start;
    max-height: calc(100vh - 5rem);
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

MODEL_COLOR = "#8c9096"  # цвет модели без текстуры

DEFAULTS = p3.Quadruped().to_dict()

PHOTOS = {  # ракурс → (подпись, демо-фото, камера по умолчанию)
    "front": ("Под углом спереди", SAMPLES / "cat_front34.png",
              json.loads((SAMPLES / "cat_front34.camera.json").read_text())),
    "back": ("Под углом сзади", SAMPLES / "cat_back34.png",
             json.loads((SAMPLES / "cat_back34.camera.json").read_text())),
}
CAM_SLIDERS = {  # поле → (подпись, мин, макс, шаг)
    "az": ("Поворот, °", -180.0, 180.0, 0.5), "el": ("Наклон сверху, °", -10.0, 60.0, 0.5),
    "roll": ("Крен, °", -15.0, 15.0, 0.25), "dist": ("Удалённость (перспектива)", 1.5, 30.0, 0.1),
    "scale": ("Масштаб", 0.2, 1.5, 0.005), "cx": ("Сдвиг по X", 0.0, 1.0, 0.005),
    "cy": ("Сдвиг по Y", 0.0, 1.0, 0.005),
}


def set_params(values: dict):
    for k, v in values.items():
        if k in DEFAULTS:
            st.session_state[f"p_{k}"] = float(v)


def load_params_file():
    f = st.session_state.get("params_file")
    if f is not None:
        set_params(json.loads(f.getvalue()))


def set_camera(view: str, values: dict):
    for k, v in values.items():
        st.session_state[f"c_{view}_{k}"] = float(v)


if "p_body_w" not in st.session_state:
    set_params(DEFAULTS)
    for view, (_, _, cam) in PHOTOS.items():
        set_camera(view, cam)


@st.cache_data(show_spinner=False, max_entries=8)
def photo_rgb(data: bytes, max_side: int = 900) -> np.ndarray:
    """Фото, уменьшенное для быстрого наложения каркаса."""
    rgb = load_image(data)[:, :, :3]
    k = max_side / max(rgb.shape[:2])
    if k < 1:
        rgb = cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return rgb


@st.cache_data(show_spinner="Переношу цвета с фото…", max_entries=4)
def baked(params_json: str, convex: bool, photos: tuple[tuple[bytes, tuple], ...]):
    """Текстура-атлас по фото и камерам (в единицах шаблона, до масштабирования)."""
    q = p3.Quadruped.from_dict(json.loads(params_json))
    parts = q.build()
    faces = {n: p3.merge_faces(m, convex) for n, m in parts.items()}
    return bake(parts, faces, [(photo_rgb(data, 1600), Camera(**dict(cam))) for data, cam in photos])


left, center, right = st.columns([1, 2, 1.2])

with right:
    st.subheader("Параметры")

    height = st.number_input("Высота модели, см", 1.0, 100.0, 15.0, step=1.0)
    convex = st.checkbox("Только выпуклые грани", value=True,
                         help="Вогнутые грани делятся на выпуклые части: не все программы развёртки "
                              "правильно читают вогнутые многоугольники из OBJ. Новые рёбра плоские (сгиб 0°).")

    for group, group_fields in p3.param_groups().items():
        with st.expander(group, expanded=group == "Тело"):
            for f in group_fields:
                m = f.metadata
                st.slider(m["label"], float(m["lo"]), float(m["hi"]), step=float(m["step"]),
                          key=f"p_{f.name}")

    params = p3.Quadruped(**{f.name: st.session_state[f"p_{f.name}"] for f in fields(p3.Quadruped)})

    c1, c2 = st.columns(2)
    c1.download_button("💾 Сохранить", json.dumps(params.to_dict(), ensure_ascii=False, indent=2),
                       file_name="paper3d_params.json", mime="application/json", width="stretch")
    c2.button("↺ Сбросить", width="stretch", on_click=set_params, args=(DEFAULTS,))
    st.file_uploader("Загрузить параметры (.json)", type=["json"], key="params_file",
                     on_change=load_params_file)

with left:
    st.subheader("Референсы")
    photos = []  # (фото, камера) — для переноса цветов
    for view, (label, demo, _) in PHOTOS.items():
        f = st.file_uploader(label, type=["png", "jpg", "jpeg"], key=f"photo_{view}")
        data = f.getvalue() if f else demo.read_bytes()
        cam = Camera(**{k: st.session_state[f"c_{view}_{k}"] for k in CAM_SLIDERS})
        photos.append((data, tuple(cam.to_dict().items())))
        parts = params.build()
        faces = {n: p3.merge_faces(m) for n, m in parts.items()}
        show = st.toggle("Каркас поверх фото", value=True, key=f"ov_{view}")
        img = photo_rgb(data)
        st.image(overlay(img, parts, faces, cam, color=(255, 30, 30)) if show else img, width="stretch")
        with st.expander("Камера", expanded=False):
            for k, (lbl, lo, hi, step) in CAM_SLIDERS.items():
                st.slider(lbl, lo, hi, step=step, key=f"c_{view}_{k}")
            st.button("↺ Камера по умолчанию", key=f"camreset_{view}", on_click=set_camera,
                      args=(view, PHOTOS[view][2]))
    st.caption("Подгоните камеру так, чтобы красный каркас лёг на линии сгиба на фото: "
               "по камерам цвета с фото переносятся на грани модели.")

with center, st.container(key="viewer"):
    st.subheader("3D-просмотр")
    try:
        model = p3.generate(params, height=height, convex=convex)
    except Exception as e:
        st.error(f"Не удалось построить модель: {e}")
        st.stop()

    t1, t2 = st.columns(2)
    colored = t1.toggle("🎨 Цвета с фото", value=True, key="colored")
    edges = t2.checkbox("Линии сгиба", value=True, key="edges")
    tex = baked(json.dumps(params.to_dict()), convex, tuple(photos)) if colored else None
    st.iframe(viewer_html(model.parts, model.faces, tex, edges=edges, height=600, color=MODEL_COLOR), height=600)
    if tex is not None:
        st.caption(f"С фото взято {tex.coverage:.0%} поверхности, остальное — по симметрии и соседним цветам.")

    lo, hi = model.bounds
    size = hi - lo
    c1, c2, c3 = st.columns(3)
    c1.metric("Деталей", len(model.parts))
    c2.metric("Граней", model.face_count)
    c3.metric("Ш × В × Д, см", f"{size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f}")

    def colored_zip(params_json=json.dumps(params.to_dict()), photos=tuple(photos)) -> bytes:
        # собирается по нажатию: цвета считаются, даже если в просмотре они выключены
        return p3.to_zip(export_textured(model.parts, model.faces, baked(params_json, convex, photos)))

    c1, c2 = st.columns(2)
    c1.download_button("⬇ OBJ без цветов", data=p3.export_obj(model.parts, model.faces),
                       file_name="papercraft_model.obj", mime="text/plain", width="stretch",
                       help="Только форма: один файл .obj")
    c2.download_button("🎨 OBJ с цветами (zip)", data=colored_zip,
                       file_name="papercraft_model_color.zip", mime="application/zip", width="stretch",
                       help="Архив из трёх файлов: .obj с UV-координатами, материал .mtl и текстура .png. "
                            "Распакуйте их в одну папку и открывайте .obj")
    if tex is not None:
        with st.expander("Текстура (атлас)"):
            st.image(tex.image, width="stretch")
