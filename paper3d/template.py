"""Шаблон «четвероногое»: тело, голова, уши, 4 лапы, хвост.

Значения по умолчанию подогнаны по точкам на референсе samples/cat_front34.png.

Размеры — в условных единицах; готовая модель масштабируется под нужную высоту.
Перед модели смотрит в +Z, низ лап — на Y = 0.
"""

from dataclasses import asdict, dataclass, field, fields

import trimesh

import numpy as np

from .primitives import chamfered_box, clip, convex_hull, rotate_x


def p(default, lo, hi, label, group, step=0.1):
    return field(default=default, metadata=dict(lo=lo, hi=hi, label=label, group=group, step=step))


@dataclass
class Quadruped:
    # тело
    body_w: float = p(6.9, 2, 15, "Ширина", "Тело")
    body_h: float = p(4.0, 2, 15, "Высота", "Тело")
    body_d: float = p(8.9, 2, 20, "Длина", "Тело")
    body_top_chamfer: float = p(0.0, 0, 3, "Скос верхних рёбер", "Тело")
    body_bottom_chamfer: float = p(0.0, 0, 3, "Скос нижних рёбер", "Тело")
    # голова
    head_w: float = p(7.3, 2, 15, "Ширина", "Голова")
    head_h: float = p(5.9, 2, 15, "Высота", "Голова")
    head_d: float = p(3.8, 2, 15, "Глубина", "Голова")
    head_y: float = p(3.3, -3, 10, "Подъём над низом тела", "Голова")
    head_out: float = p(1.2, -5, 8, "Выступ вперёд за тело", "Голова")
    head_top_side: float = p(1.9, 0.3, 4, "Скос: верх по бокам (под ушами)", "Голова")
    head_forehead: float = p(0.5, 0, 3, "Лоб: наклон назад", "Голова")
    head_nape: float = p(1.5, 0, 6, "Затылок: длина скоса", "Голова")
    head_bottom_side: float = p(0.8, 0, 3, "Скос: щёки", "Голова")
    head_bottom_front: float = p(0.0, 0, 3, "Скос: подбородок", "Голова")
    # уши: четырёхгранники на верхних боковых скосах головы
    ear_h: float = p(1.3, 0.5, 6, "Высота", "Уши")
    ear_lean: float = p(0.1, -2, 2, "Наклон кончика наружу", "Уши")
    # лапы
    leg_h: float = p(2.9, 0.5, 12, "Высота", "Лапы")
    leg_w: float = p(2.3, 0.5, 6, "Ширина", "Лапы")
    leg_d: float = p(2.1, 0.5, 6, "Глубина", "Лапы")
    leg_inset_x: float = p(0.0, 0, 3, "Отступ от боков тела", "Лапы")
    leg_inset_z: float = p(0.0, 0, 5, "Отступ от переда/зада тела", "Лапы")
    paw_len: float = p(0.8, 0, 3, "Подушечка: вынос вперёд", "Лапы")
    paw_h: float = p(0.9, 0, 3, "Подушечка: высота", "Лапы")
    # хвост: по центру, прижат к задней стенке тела
    tail_w: float = p(1.6, 0.3, 4, "Ширина", "Хвост")
    tail_d: float = p(1.5, 0.3, 4, "Толщина", "Хвост")
    tail_len: float = p(5.0, 0, 15, "Длина", "Хвост")
    tail_y: float = p(4.0, 0, 10, "Высота крепления над низом тела", "Хвост")
    tail_tilt: float = p(35.0, 0, 80, "Наклон назад, °", "Хвост", step=1.0)
    tail_tip: float = p(1.1, 0, 2, "Скос кончика (конёк)", "Хвост")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Quadruped":
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in names})

    def tail_base(self) -> float:
        """Высота верхнего края среза хвоста: срез не выходит за низ и верх задней стенки."""
        y0, y1 = self.leg_h, self.leg_h + self.body_h
        tilt = np.radians(max(self.tail_tilt, 1e-3))
        cut = min(self.tail_d / np.sin(tilt), self.body_h)  # высота площадки среза
        return float(np.clip(y0 + self.tail_y, y0 + cut, y1))

    def build(self) -> dict[str, trimesh.Trimesh]:
        """Отдельные замкнутые детали модели."""
        parts = {}
        y0, y1 = self.leg_h, self.leg_h + self.body_h
        zf, zb = self.body_d / 2, -self.body_d / 2
        bw = self.body_w / 2

        ct, cb = self.body_top_chamfer, self.body_bottom_chamfer
        parts["body"] = chamfered_box(
            (-bw, y0, zb), (bw, y1, zf),
            {e: ct for e in ("top+front", "top+back", "top+left", "top+right")}
            | {e: cb for e in ("bottom+front", "bottom+back", "bottom+left", "bottom+right")})

        # голова
        hy0 = y0 + self.head_y
        hy1 = hy0 + self.head_h
        hz1 = zf + self.head_out
        hz0 = hz1 - self.head_d
        hw = self.head_w / 2
        # голова: морда — прямоугольник со скосами щёк снизу; над линией бровей (на высоту
        # скоса c) лоб наклонён назад на f. Сверху ровная площадка, за ней скос затылка
        # опускается на c на длине n — затылок прямоугольный, боковые скосы под ушами
        # сходятся к его верхним углам
        c = min(self.head_top_side, 0.5 * min(self.head_w, self.head_h))
        f = min(self.head_forehead, 0.9 * self.head_d)
        n = min(max(self.head_nape, 0.01), self.head_d - f)
        hzt, hzn = hz1 - f, hz0 + n  # передний и задний край площадки
        head = chamfered_box((-hw, hy0, hz0), (hw, hy1, hz1), {
            "top+left": c, "top+right": c,
            "bottom+left": self.head_bottom_side, "bottom+right": self.head_bottom_side,
            "bottom+front": self.head_bottom_front,
        })
        head = clip(head, (0, f, c), (0, hy1 - c, hz1))     # лоб
        parts["head"] = clip(head, (0, n, -c), (0, hy1, hzn))  # затылок

        # уши: на боковом скосе головы, с коньком от лба до конца площадки. Передняя грань —
        # в плоскости лба, внешняя — в плоскости бока, задняя спускается к углу затылка
        az = hz1 - f * (c + self.ear_h) / c
        for side, sx in (("left", -1), ("right", 1)):
            base = [(sx * (hw - c), hy1, hzt), (sx * hw, hy1 - c, hz1),
                    (sx * hw, hy1 - c, hz0), (sx * (hw - c), hy1, hzn)]
            ridge = [(sx * (hw + self.ear_lean), hy1 + self.ear_h, z) for z in {az, min(hzn, az)}]
            parts[f"ear_{side}"] = convex_hull(base + ridge)

        # лапы: столбик + отдельная подушечка спереди (приклеивается к лапе)
        lx = bw - self.leg_inset_x - self.leg_w / 2
        for zname, lz in (("front", zf - self.leg_inset_z - self.leg_d / 2),
                          ("back", zb + self.leg_inset_z + self.leg_d / 2)):
            for side, sx in (("left", -1), ("right", 1)):
                x = sx * lx
                lo = (x - self.leg_w / 2, 0, lz - self.leg_d / 2)
                hi = (x + self.leg_w / 2, y0, lz + self.leg_d / 2)
                parts[f"leg_{zname}_{side}"] = chamfered_box(lo, hi)
                if self.paw_len > 0 and self.paw_h > 0:
                    parts[f"paw_{zname}_{side}"] = chamfered_box(
                        (lo[0], 0, hi[2]), (hi[0], min(self.paw_h, y0), hi[2] + self.paw_len),
                        {"top+front": min(self.paw_h, self.paw_len) * 0.6})

        # хвост: наклонный брусок по центру задней стенки тела. Весь низ хвоста срезан
        # вертикально и целиком прилегает к стенке; верхний край среза — на высоте tail_y.
        # Кончик — конёк поперёк хвоста.
        if self.tail_len > 0:
            base = self.tail_base()
            t = self.tail_tip
            tilt = np.radians(self.tail_tilt)
            ext = min(self.tail_d / np.tan(tilt), self.tail_len) if tilt > 1e-3 else 0.0  # запас под срез
            tail = chamfered_box((-self.tail_w / 2, base - ext, zb - self.tail_d),
                                 (self.tail_w / 2, base + self.tail_len, zb),
                                 {"top+front": t, "top+back": t})
            tail = rotate_x(tail, -self.tail_tilt, (0, base, zb))
            parts["tail"] = clip(tail, (0, 0, 1), (0, 0, zb))
        return parts


def keypoints(q: Quadruped) -> dict[str, tuple[float, float, float]]:
    """Характерные углы деталей — для подгонки по точкам на фото.

    «right» — правый бок модели (+X), «left» — левый (−X).
    """
    y0, y1 = q.leg_h, q.leg_h + q.body_h
    zf, zb = q.body_d / 2, -q.body_d / 2
    bw, hw = q.body_w / 2, q.head_w / 2
    hy0, hy1 = y0 + q.head_y, y0 + q.head_y + q.head_h
    hz1 = zf + q.head_out
    hz0 = hz1 - q.head_d
    ts, ch, cb = q.head_top_side, q.head_bottom_side, q.body_bottom_chamfer
    f = min(q.head_forehead, 0.9 * q.head_d)
    ex, ey, ez = hw + q.ear_lean, hy1 + q.ear_h, hz1 - f * (ts + q.ear_h) / ts
    lx = bw - q.leg_inset_x - q.leg_w / 2
    lzf = zf - q.leg_inset_z                    # перед передних лап
    lzb = zb + q.leg_inset_z + q.leg_d          # перед задних лап
    tz0, tz1 = zb - q.tail_d, zb
    tx0, tx1 = -q.tail_w / 2, q.tail_w / 2
    top = y0 + q.tail_y + q.tail_len
    tt = min(q.tail_tip, q.tail_d / 2)
    ridge_z = zb - q.tail_d / 2 if q.tail_tip >= q.tail_d / 2 else tz1 - tt  # передний край верха
    kp = {
        "head_front_top_left": (-hw + ts, hy1, hz1 - f),
        "head_front_top_right": (hw - ts, hy1, hz1 - f),
        "head_front_side_top_left": (-hw, hy1 - ts, hz1),
        "head_front_side_top_right": (hw, hy1 - ts, hz1),
        "head_front_cheek_left": (-hw, hy0 + ch, hz1),
        "head_front_cheek_right": (hw, hy0 + ch, hz1),
        "head_front_bottom_left": (-hw + ch, hy0, hz1),
        "head_front_bottom_right": (hw - ch, hy0, hz1),
        "head_back_side_top_right": (hw, hy1 - ts, hz0),
        "head_back_cheek_right": (hw, hy0 + ch, hz0),
        "ear_tip_left": (-ex, ey, ez),
        "ear_tip_right": (ex, ey, ez),
        "body_front_bottom_left": (-bw, y0 + cb, zf),
        "body_front_bottom_right": (bw, y0 + cb, zf),
        "body_back_bottom_right": (bw, y0 + cb, zb),
        "body_back_top_right": (bw, y1, zb),
        "body_back_top_left": (-bw, y1, zb),
        "body_back_bottom_left": (-bw, y0 + cb, zb),
        "body_front_bottom_left_side": (-bw, y0 + cb, zf),
        "head_back_side_top_left": (-hw, hy1 - ts, hz0),
        "head_back_cheek_left": (-hw, hy0 + ch, hz0),
        "leg_front_left_top_inner": (-lx + q.leg_w / 2, y0, lzf),
        "leg_front_right_top_inner": (lx - q.leg_w / 2, y0, lzf),
        "leg_front_left_paw_top": (-lx - q.leg_w / 2, q.paw_h, lzf),
        "paw_front_left_outer": (-lx - q.leg_w / 2, 0, lzf + q.paw_len),
        "paw_front_left_inner": (-lx + q.leg_w / 2, 0, lzf + q.paw_len),
        "paw_front_right_inner": (lx - q.leg_w / 2, 0, lzf + q.paw_len),
        "paw_front_right_outer": (lx + q.leg_w / 2, 0, lzf + q.paw_len),
        "paw_back_right_inner": (lx - q.leg_w / 2, 0, lzb + q.paw_len),
        "paw_back_right_outer": (lx + q.leg_w / 2, 0, lzb + q.paw_len),
        "tail_front_top_left": (tx0, top - tt, tz1),
        "tail_front_top_right": (tx1, top - tt, tz1),
        "tail_back_top_right": (tx1, top - tt, tz0),
        "tail_ridge_left": (tx0, top, ridge_z),
        "tail_ridge_right": (tx1, top, ridge_z),
    }
    # наклон хвоста: поворот вокруг верхнего края среза (0, y0 + tail_y, zb)
    a = np.radians(q.tail_tilt)
    pivot = np.array([0.0, q.tail_base(), zb])
    rot = np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]])
    for n in kp:
        if n.startswith("tail_"):
            kp[n] = tuple(pivot + rot @ (np.array(kp[n]) - pivot))
    return kp


def param_groups() -> dict[str, list]:
    """Поля шаблона, сгруппированные для интерфейса."""
    groups: dict[str, list] = {}
    for f in fields(Quadruped):
        groups.setdefault(f.metadata["group"], []).append(f)
    return groups
