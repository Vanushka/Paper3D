"""Отделение объекта от фона на фото."""

import cv2
import numpy as np


def load_image(data: bytes) -> np.ndarray:
    """Декодирует файл в RGBA."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Не удалось прочитать изображение")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def extract_mask(rgba: np.ndarray, method: str = "auto", threshold: int = 30) -> np.ndarray:
    """Отделяет объект от фона.

    auto    — прозрачность, если она есть, иначе отличие от цвета фона по краям кадра;
    grabcut — для неоднородного фона (медленнее).
    """
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]

    if method == "auto" and alpha.min() < 250:
        mask = (alpha > 127).astype(np.uint8) * 255
    elif method == "grabcut":
        h, w = alpha.shape
        m = max(2, int(0.02 * min(h, w)))
        gc_mask = np.zeros((h, w), np.uint8)
        bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.grabCut(bgr, gc_mask, (m, m, w - 2 * m, h - 2 * m), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
    else:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(lab - bg, axis=2)
        mask = (dist > threshold).astype(np.uint8) * 255

    return _clean_mask(mask)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Убирает шум, оставляет самый крупный объект и заливает дыры."""
    k = max(3, int(0.005 * max(mask.shape)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Объект на изображении не найден")
    biggest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [biggest], -1, 255, cv2.FILLED)
    return out


def fill_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Заменяет фон ближайшим пикселем объекта, чтобы фон не попадал в текстуру."""
    _, labels = cv2.distanceTransformWithLabels(
        (mask == 0).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    obj = np.argwhere(mask > 0)  # метки объекта идут в построчном порядке
    nearest = obj[labels - 1]
    return image[nearest[..., 0], nearest[..., 1]]
