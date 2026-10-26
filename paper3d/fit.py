"""Подгонка ракурса камеры и размеров шаблона под силуэт объекта на фото."""

from dataclasses import fields, replace

import cv2
import numpy as np
from scipy.optimize import minimize

from .camera import Camera, silhouette
from .template import Quadruped

# границы параметров камеры для оптимизации
CAM_BOUNDS = {
    "az": (-180, 180), "el": (-10, 60), "roll": (-15, 15), "dist": (1.5, 30),
    "scale": (0.2, 1.5), "cx": (0.0, 1.0), "cy": (0.0, 1.0),
}


def _prep_mask(mask: np.ndarray, size: int):
    h, w = mask.shape
    k = size / max(h, w)
    small = cv2.resize(mask, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255


def _soft(mask: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(mask, (0, 0), sigma)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    return float((a & b).sum() / max(1, (a | b).sum()))


def init_camera(ref_mask: np.ndarray, params: Quadruped, az: float) -> Camera:
    """Начальный ракурс: совмещаем рамку силуэта модели и фото."""
    h, w = ref_mask.shape
    ys, xs = np.nonzero(ref_mask > 0.5)
    cam = Camera(az=az)
    parts = params.build()
    m = silhouette(parts, cam, w, h)
    my, mx = np.nonzero(m)
    cam.scale *= (ys.max() - ys.min()) / max(1, my.max() - my.min())
    m = silhouette(parts, cam, w, h)
    my, mx = np.nonzero(m)
    cam.cx += ((xs.min() + xs.max()) - (mx.min() + mx.max())) / 2 / w
    cam.cy += ((ys.min() + ys.max()) - (my.min() + my.max())) / 2 / h
    return cam


def fit(ref_mask: np.ndarray, params: Quadruped, cam: Camera | None = None, *,
        az_guess: float = 30.0, fit_shape: bool = True, size: int = 192,
        iters: int = 3000, fixed: tuple[str, ...] = ()) -> tuple[Quadruped, Camera, float]:
    """Подбирает камеру (и размеры деталей при fit_shape) по маске объекта.

    Возвращает (параметры шаблона, камера, IoU силуэтов).
    """
    ref = _prep_mask(ref_mask, size)
    h, w = ref.shape
    if cam is None:
        cam = init_camera(ref, params, az_guess)

    shape_fields = [f for f in fields(Quadruped) if fit_shape and f.name not in fixed]
    cam_names = list(CAM_BOUNDS)
    lo = np.array([CAM_BOUNDS[n][0] for n in cam_names] + [f.metadata["lo"] for f in shape_fields], float)
    hi = np.array([CAM_BOUNDS[n][1] for n in cam_names] + [f.metadata["hi"] for f in shape_fields], float)
    x0 = np.array([getattr(cam, n) for n in cam_names] + [getattr(params, f.name) for f in shape_fields], float)
    span = hi - lo

    def unpack(u):
        x = np.clip(lo + u * span, lo, hi)
        c = replace(cam, **dict(zip(cam_names, x[:len(cam_names)])))
        q = replace(params, **{f.name: float(v) for f, v in zip(shape_fields, x[len(cam_names):])})
        return q, c

    def loss(u, sigma):
        q, c = unpack(u)
        try:
            m = silhouette(q.build(), c, w, h).astype(np.float32) / 255
        except Exception:
            return 2.0
        a, b = _soft(m, sigma), _soft(ref, sigma)
        soft_iou = (a * b).sum() / max(1e-6, (a + b - a * b).sum())
        return 1 - soft_iou

    u = (x0 - lo) / span
    # сначала только камера, потом всё вместе; от размытого к чёткому силуэту
    n_cam = len(cam_names)
    for sigma, free in ((3.0, n_cam), (2.0, len(u)), (1.0, len(u))):
        def sub_loss(v, sigma=sigma, free=free):
            uu = u.copy()
            uu[:free] = v
            return loss(uu, sigma)
        res = minimize(sub_loss, u[:free], method="Powell",
                       bounds=[(0, 1)] * free, options=dict(maxfev=iters, xtol=1e-3, ftol=1e-4))
        u[:free] = res.x

    q, c = unpack(u)
    return q, c, iou(silhouette(q.build(), c, w, h) / 255, ref)


def fit_keypoints(points: dict[str, tuple[float, float]], w: int, h: int, params: Quadruped,
                  cam: Camera, prior: float = 0.01,
                  fixed: tuple[str, ...] = ()) -> tuple[Quadruped, Camera, float]:
    """Подбирает камеру и размеры так, чтобы углы деталей легли на отмеченные точки фото.

    points — {имя ключевой точки: (x, y) в пикселях}. Параметры, которые точками
    не определяются, остаются около исходных значений (слабая «пружина» prior).
    Возвращает (параметры, камера, средняя ошибка в пикселях).
    """
    from scipy.optimize import least_squares
    from .camera import model_frame
    from .template import keypoints

    names = [n for n in points if n in keypoints(params)]
    obs = np.array([points[n] for n in names], float)
    shape_fields = [f for f in fields(Quadruped) if f.name not in fixed]
    cam_names = list(CAM_BOUNDS)
    lo = np.array([CAM_BOUNDS[n][0] for n in cam_names] + [f.metadata["lo"] for f in shape_fields], float)
    hi = np.array([CAM_BOUNDS[n][1] for n in cam_names] + [f.metadata["hi"] for f in shape_fields], float)
    x0 = np.array([getattr(cam, n) for n in cam_names] + [getattr(params, f.name) for f in shape_fields], float)
    x0 = np.clip(x0, lo, hi)
    span = hi - lo
    n_cam = len(cam_names)

    def unpack(x):
        c = replace(cam, **dict(zip(cam_names, x[:n_cam])))
        q = replace(params, **{f.name: float(v) for f, v in zip(shape_fields, x[n_cam:])})
        return q, c

    def residuals(x):
        q, c = unpack(x)
        kp = keypoints(q)
        # опорная точка и высота — как у собранной модели (без сборки — по габаритам ключевых точек)
        target, height = model_frame(q.build())
        uv, _ = c.project(np.array([kp[n] for n in names]), target, height, w, h)
        r = ((uv - obs) / h).ravel()
        return np.concatenate([r, prior * (x[n_cam:] - x0[n_cam:]) / span[n_cam:]])

    res = least_squares(residuals, x0, bounds=(lo, hi), x_scale=span, diff_step=1e-4)
    q, c = unpack(res.x)
    err = np.sqrt((res.fun[:2 * len(names)].reshape(-1, 2) ** 2).sum(1)).mean() * h
    return q, c, float(err)
