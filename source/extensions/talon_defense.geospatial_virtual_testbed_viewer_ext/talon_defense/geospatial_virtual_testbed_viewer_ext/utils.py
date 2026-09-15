import math
import numpy as np
from pxr import Gf

from datetime import timedelta


from .constants import *

# Euclidean distance
def distance(a, b) -> float:
    assert(len(a) == len(b))

    v = 0.
    for x in a:
        for y in b:
            v += math.pow(x - y, 2)

    return math.sqrt(v)

def to_vec3d(x) -> Gf.Vec3d:
    return Gf.Vec3d(x[0], x[1], x[2])

def to_vec3f(x) -> Gf.Vec3f:
    return Gf.Vec3f(x[0], x[1], x[2])

def xyz_to_lla(x, y, z):
    '''ECEF coordinates (in kilometers) to geodetic (WGS84 ellipsoid)'''

    a = WGS84_SEMIMAJOR
    b = WGS84_SEMIMINOR

    f = (a - b) / a
    f_inv = 1.0 / f

    e_sq = f * (2 - f)
    eps = e_sq / (1.0 - e_sq)

    p = math.sqrt(x * x + y * y)
    q = math.atan2((z * a), (p * b))

    sin_q = math.sin(q)
    cos_q = math.cos(q)

    sin_q_3 = sin_q * sin_q * sin_q
    cos_q_3 = cos_q * cos_q * cos_q

    phi = math.atan2((z + eps * b * sin_q_3), (p - e_sq * a * cos_q_3))
    lam = math.atan2(y, x)

    v = a / math.sqrt(1.0 - e_sq * math.sin(phi) * math.sin(phi))
    h   = (p / math.cos(phi)) - v

    lat = math.degrees(phi)
    lon = math.degrees(lam)

    return (lat,lon,h)

def xyz_to_spherical(x, y, z) -> tuple[float, float, float]:
     r = math.sqrt(x**2 + y**2 + z**2)
     theta = math.acos(z / r)
     phi = math.atan(y / x)

     return (r, math.degrees(theta), math.degrees(phi))