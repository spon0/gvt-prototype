"""Value -> (color, opacity) transfer functions. numpy only; no matplotlib needed in Kit."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# 17 evenly spaced sRGB control points sampled from matplotlib's colormaps
# (viridis/magma: CC0, turbo: Apache-2.0, coolwarm/Blues: matplotlib license).
# Linear interpolation between them is within ~0.03 of the full 256-entry tables.
_CONTROL_POINTS: dict[str, list[tuple[float, float, float]]] = {
    "viridis": [
        (0.2670, 0.0049, 0.3294), (0.2823, 0.0950, 0.4173), (0.2788, 0.1755, 0.4834), (0.2590, 0.2515, 0.5247),
        (0.2297, 0.3224, 0.5457), (0.1994, 0.3876, 0.5546), (0.1727, 0.4488, 0.5579), (0.1490, 0.5081, 0.5573),
        (0.1276, 0.5669, 0.5506), (0.1206, 0.6258, 0.5335), (0.1579, 0.6838, 0.5017), (0.2461, 0.7389, 0.4520),
        (0.3692, 0.7889, 0.3829), (0.5160, 0.8312, 0.2943), (0.6785, 0.8637, 0.1895), (0.8456, 0.8873, 0.0997),
        (0.9932, 0.9062, 0.1439),
    ],
    "turbo": [
        (0.1900, 0.0718, 0.2322), (0.2511, 0.2524, 0.6337), (0.2763, 0.4212, 0.8912), (0.2586, 0.5796, 0.9988),
        (0.1584, 0.7355, 0.9231), (0.0927, 0.8655, 0.7623), (0.1966, 0.9490, 0.5947), (0.4278, 0.9942, 0.3857),
        (0.6436, 0.9900, 0.2336), (0.8047, 0.9245, 0.2046), (0.9330, 0.8124, 0.2267), (0.9931, 0.6741, 0.2035),
        (0.9836, 0.4929, 0.1285), (0.9211, 0.3149, 0.0548), (0.8161, 0.1846, 0.0181), (0.6645, 0.0844, 0.0042),
        (0.4796, 0.0158, 0.0106),
    ],
    "magma": [
        (0.0015, 0.0005, 0.0139), (0.0396, 0.0311, 0.1335), (0.1131, 0.0655, 0.2768), (0.2117, 0.0620, 0.4186),
        (0.3167, 0.0717, 0.4854), (0.4147, 0.1104, 0.5047), (0.5128, 0.1482, 0.5076), (0.6136, 0.1818, 0.4985),
        (0.7164, 0.2150, 0.4753), (0.8169, 0.2559, 0.4365), (0.9043, 0.3196, 0.3881), (0.9609, 0.4183, 0.3596),
        (0.9867, 0.5356, 0.3822), (0.9961, 0.6537, 0.4462), (0.9969, 0.7696, 0.5349), (0.9924, 0.8843, 0.6401),
        (0.9871, 0.9914, 0.7495),
    ],
    "coolwarm": [
        (0.2298, 0.2987, 0.7537), (0.3042, 0.4069, 0.8453), (0.3837, 0.5102, 0.9178), (0.4677, 0.6056, 0.9685),
        (0.5543, 0.6901, 0.9955), (0.6408, 0.7608, 0.9978), (0.7240, 0.8149, 0.9757), (0.8006, 0.8504, 0.9300),
        (0.8674, 0.8644, 0.8626), (0.9256, 0.8255, 0.7711), (0.9595, 0.7670, 0.6741), (0.9697, 0.6905, 0.5751),
        (0.9567, 0.5980, 0.4773), (0.9214, 0.4914, 0.3834), (0.8654, 0.3711, 0.2958), (0.7906, 0.2314, 0.2162),
        (0.7057, 0.0156, 0.1502),
    ],
    "blues": [
        (0.9686, 0.9843, 1.0000), (0.9194, 0.9528, 0.9843), (0.8702, 0.9213, 0.9685), (0.8230, 0.8898, 0.9528),
        (0.7752, 0.8583, 0.9368), (0.6965, 0.8248, 0.9093), (0.6173, 0.7909, 0.8818), (0.5169, 0.7357, 0.8602),
        (0.4171, 0.6806, 0.8382), (0.3364, 0.6255, 0.8067), (0.2563, 0.5700, 0.7752), (0.1913, 0.5051, 0.7417),
        (0.1271, 0.4402, 0.7075), (0.0779, 0.3772, 0.6583), (0.0314, 0.3141, 0.6065), (0.0314, 0.2491, 0.5100),
        (0.0314, 0.1882, 0.4196),
    ],
    "gray": [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
    "white": [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
}

COLORMAPS = tuple(sorted(_CONTROL_POINTS))


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def make_lut(cmap: Any = "turbo", size: int = 256, linear: bool = True) -> np.ndarray:
    """(size, 3) float32 lookup table.

    ``cmap`` is a name from COLORMAPS (append ``_r`` to reverse), an (N, 3|4) array
    of sRGB colors in 0..1, or any callable like a matplotlib Colormap.
    Colormaps are designed in sRGB; RTX shades in linear space, so by default the
    table is linearized to make the viewport show the colormap as designed.
    """
    if isinstance(cmap, str):
        name = cmap.lower()
        reverse = name.endswith("_r")
        name = name[:-2] if reverse else name
        if name not in _CONTROL_POINTS:
            raise KeyError(f"Unknown colormap '{cmap}'. Built in: {', '.join(COLORMAPS)}")
        ctrl = np.asarray(_CONTROL_POINTS[name], dtype=np.float64)
        if reverse:
            ctrl = ctrl[::-1]
    elif callable(cmap):
        ctrl = np.asarray(cmap(np.linspace(0.0, 1.0, size)), dtype=np.float64)[:, :3]
    else:
        ctrl = np.asarray(cmap, dtype=np.float64)[:, :3]
    x_ctrl = np.linspace(0.0, 1.0, len(ctrl))
    x = np.linspace(0.0, 1.0, size)
    lut = np.stack([np.interp(x, x_ctrl, ctrl[:, k]) for k in range(3)], axis=-1)
    if linear:
        lut = srgb_to_linear(lut)
    return lut.astype(np.float32)


@dataclass
class OpacityRamp:
    """Opacity rises from ``a0`` at value ``v0`` to ``a1`` at ``v1`` (data units, clamped).

    Example: precipitable water, clear below 20 kg/m^2, 85% at 60 kg/m^2:
    ``OpacityRamp(20, 60, 0.0, 0.85)``. Use v0 > v1 for "more opaque when lower".
    """

    v0: float
    v1: float
    a0: float = 0.0
    a1: float = 1.0
    gamma: float = 1.0

    def __call__(self, values: np.ndarray) -> np.ndarray:
        span = (self.v1 - self.v0) or 1e-12
        t = np.clip((values - self.v0) / span, 0.0, 1.0)
        if self.gamma != 1.0:
            t = t ** self.gamma
        return self.a0 + (self.a1 - self.a0) * t


@dataclass
class Style:
    """How field values become vertex colors and opacities.

    opacity: a constant, an :class:`OpacityRamp`, or any ``f(values) -> alpha`` callable.
    vmin/vmax: color range in data units; ``None`` = fill from the first frame
    (2nd/98th percentile) and then keep it fixed so colors stay stable over time.
    """

    cmap: Any = "turbo"
    vmin: float | None = None
    vmax: float | None = None
    opacity: float | Callable[[np.ndarray], np.ndarray] = 0.6
    missing_opacity: float = 0.0
    log_scale: bool = False  # color by log10(value), e.g. precipitation rate
    linear_color: bool = True
    lut_size: int = 256
    _lut: np.ndarray | None = field(default=None, init=False, repr=False)
    _lut_key: Any = field(default=None, init=False, repr=False)

    @property
    def lut(self) -> np.ndarray:
        key = (id(self.cmap), self.cmap if isinstance(self.cmap, str) else None,
               self.lut_size, self.linear_color)
        if self._lut is None or key != self._lut_key:
            self._lut = make_lut(self.cmap, self.lut_size, self.linear_color)
            self._lut_key = key
        return self._lut

    def _transform(self, v: np.ndarray) -> np.ndarray:
        if not self.log_scale:
            return v
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(np.maximum(v, 1e-12))

    def autoscale(self, values: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> "Style":
        if self.vmin is None or self.vmax is None:
            v = self._transform(np.asarray(values, dtype=np.float64))
            v = v[np.isfinite(v)]
            if v.size:
                lo, hi = np.percentile(v, [lo_pct, hi_pct])
                if hi <= lo:
                    hi = lo + 1.0
                if self.log_scale:
                    lo, hi = 10.0 ** lo, 10.0 ** hi
                self.vmin = float(lo) if self.vmin is None else self.vmin
                self.vmax = float(hi) if self.vmax is None else self.vmax
        return self

    def colorize(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (rgb (N,3) float32, alpha (N,) float32)."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if self.vmin is None or self.vmax is None:
            self.autoscale(values)
        lut = self.lut
        lo, hi = self._transform(np.array([self.vmin, self.vmax], dtype=np.float64))
        scale = (len(lut) - 1) / ((hi - lo) or 1e-12)
        t = (self._transform(values) - np.float32(lo)) * np.float32(scale)
        finite = np.isfinite(t)
        idx = np.clip(np.nan_to_num(t, nan=0.0, posinf=len(lut) - 1, neginf=0.0) + 0.5,
                      0, len(lut) - 1).astype(np.int32)
        rgb = lut[idx]
        if callable(self.opacity):
            with np.errstate(invalid="ignore"):
                alpha = np.asarray(self.opacity(values), dtype=np.float32)
            alpha = np.broadcast_to(alpha, values.shape)
        else:
            alpha = np.full(values.shape, float(self.opacity), dtype=np.float32)
        alpha = np.where(finite, np.clip(alpha, 0.0, 1.0), np.float32(self.missing_opacity))
        return rgb, alpha.astype(np.float32, copy=False)
