"""
Extracción de keyframes de un clip de vídeo.

Lee el vídeo de forma secuencial (más fiable que saltar con CAP_PROP_POS_FRAMES,
que falla con algunos códecs) y se queda solo con los frames pedidos.

Dos modos:
    - uniforme: N frames repartidos por todo el clip (Etapa 1, sin pose).
    - ventana:  N frames alrededor de un frame central, p. ej. el contacto
                detectado por el disparador (Etapa 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
from PIL import Image

LADO_MAX = 640  # px del lado largo: controla cuántos tokens consume cada imagen


@dataclass
class Keyframe:
    imagen: Image.Image
    indice: int           # nº de frame en el clip
    segundo: float        # tiempo en el clip (tiempo de vídeo, no real si es cámara lenta)
    vista: str            # nombre del fichero de origen


def info_video(ruta: Path) -> tuple[int, float]:
    """Devuelve (nº de frames, fps)."""
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir {ruta}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return total, fps


def indices_uniformes(total: int, n: int) -> List[int]:
    """n índices repartidos uniformemente, sin incluir los extremos exactos."""
    if total <= 0 or n <= 0:
        return []
    n = min(n, total)
    paso = total / n
    return [int(paso * i + paso / 2) for i in range(n)]


def indices_ventana(total: int, n: int, centro: int, antes: int, despues: int) -> List[int]:
    """n índices uniformes en [centro - antes, centro + despues], recortado al clip.

    El frame central se incluye siempre.
    """
    ini = max(0, centro - antes)
    fin = min(total - 1, centro + despues)
    if fin <= ini or n <= 1:
        return [min(max(centro, 0), total - 1)]
    paso = (fin - ini) / (n - 1)
    idx = sorted({int(round(ini + paso * i)) for i in range(n)} | {centro})
    return [i for i in idx if 0 <= i < total]


def _redimensionar(frame_bgr, lado_max: int) -> Image.Image:
    h, w = frame_bgr.shape[:2]
    escala = min(1.0, lado_max / max(h, w))
    if escala < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * escala), int(h * escala)),
                               interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def leer_frames(ruta: Path, indices: Sequence[int], lado_max: int = LADO_MAX) -> List[Keyframe]:
    """Lee los frames indicados de un vídeo."""
    ruta = Path(ruta)
    _, fps = info_video(ruta)
    pendientes = set(indices)
    salida: List[Keyframe] = []
    cap = cv2.VideoCapture(str(ruta))
    i = 0
    while pendientes:
        ok, frame = cap.read()
        if not ok:
            break
        if i in pendientes:
            salida.append(Keyframe(_redimensionar(frame, lado_max), i, i / fps, ruta.name))
            pendientes.discard(i)
        i += 1
    cap.release()
    return salida


def extraer_keyframes(
    ruta: Path,
    n: int = 6,
    centro: Optional[int] = None,
    antes: int = 20,
    despues: int = 50,
    lado_max: int = LADO_MAX,
) -> List[Keyframe]:
    """Extrae n keyframes de un clip.

    Sin `centro`: reparto uniforme por todo el clip.
    Con `centro`: ventana [centro - antes, centro + despues]. Los valores por
    defecto (20 / 50 frames a 25 fps) son ~0,8 s antes y ~2 s después, para ver
    la aproximación y la posible ventaja.
    """
    total, _ = info_video(ruta)
    if centro is None:
        idx = indices_uniformes(total, n)
    else:
        idx = indices_ventana(total, n, centro, antes, despues)
    return leer_frames(ruta, idx, lado_max)


def keyframes_multivista(
    rutas: Sequence[Path],
    n_por_vista: int = 4,
    max_total: int = 12,
    centros: Optional[Sequence[Optional[int]]] = None,
    **kwargs,
) -> List[Keyframe]:
    """Keyframes de varias vistas de la misma jugada.

    `centros` permite un frame de contacto distinto por vista (las vistas de
    MVFoul no están sincronizadas). Si se supera `max_total`, se descartan las
    últimas vistas.
    """
    salida: List[Keyframe] = []
    for k, ruta in enumerate(rutas):
        if len(salida) + n_por_vista > max_total:
            break
        centro = centros[k] if centros else None
        salida.extend(extraer_keyframes(ruta, n=n_por_vista, centro=centro, **kwargs))
    return salida
