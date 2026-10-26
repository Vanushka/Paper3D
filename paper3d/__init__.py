"""Paper3D: low-poly модели для papercraft из простых деталей по шаблону (без ИИ)."""

from dataclasses import dataclass

import numpy as np
import trimesh

from .export import Face, export_obj, merge_faces, to_zip
from .template import Quadruped, param_groups


@dataclass
class Model:
    parts: dict[str, trimesh.Trimesh]
    faces: dict[str, list[Face]]

    @property
    def face_count(self) -> int:
        return sum(len(f) for f in self.faces.values())

    @property
    def bounds(self) -> np.ndarray:
        return np.array([np.min([m.bounds[0] for m in self.parts.values()], axis=0),
                         np.max([m.bounds[1] for m in self.parts.values()], axis=0)])


def generate(params: Quadruped, height: float | None = None, convex: bool = True) -> Model:
    """Строит детали по шаблону; height — итоговая высота модели (в единицах OBJ)."""
    parts = params.build()
    if height:
        lo, hi = Model(parts, {}).bounds
        k = height / (hi[1] - lo[1])
        for m in parts.values():
            m.apply_scale(k)
    return Model(parts, {n: merge_faces(m, convex) for n, m in parts.items()})
