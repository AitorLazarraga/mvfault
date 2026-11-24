"""
Reproduce todas las vistas de una acción de MVFoul a la vez, en mosaico,
con sus anotaciones encima. Sirve para revisar jugadas como árbitro.

Uso:
    python eval/ver_accion.py data/mvfouls/Valid 42

Teclas:  espacio = pausa   ,  .  = frame atrás / adelante (en pausa)
         r = reiniciar     q / Esc = salir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ALTO_VISTA = 360


def cargar(split_dir: Path, accion: str):
    with open(split_dir / "annotations.json", encoding="utf-8") as f:
        anot = json.load(f)["Actions"][accion]
    carpeta = split_dir / f"action_{accion}"
    videos = sorted(carpeta.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No hay .mp4 en {carpeta}")
    return anot, videos


def leer_frames(ruta: Path):
    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        escala = ALTO_VISTA / f.shape[0]
        frames.append(cv2.resize(f, None, fx=escala, fy=escala))
    cap.release()
    return frames, fps


def mosaico(vistas, i):
    imgs = [v[min(i, len(v) - 1)] for v in vistas]
    ancho = max(im.shape[1] for im in imgs)
    imgs = [cv2.copyMakeBorder(im, 0, 0, 0, ancho - im.shape[1], cv2.BORDER_CONSTANT)
            for im in imgs]
    if len(imgs) % 2:
        imgs.append(np.zeros_like(imgs[0]))
    filas = [np.hstack(imgs[k:k + 2]) for k in range(0, len(imgs), 2)]
    return np.vstack(filas)


def texto(img, lineas, frame, total):
    y = 22
    for linea in lineas + [f"frame {frame}/{total - 1}"]:
        cv2.putText(img, linea, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, linea, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22


def main():
    p = argparse.ArgumentParser()
    p.add_argument("split_dir", type=Path)
    p.add_argument("accion")
    args = p.parse_args()

    anot, rutas = cargar(args.split_dir, args.accion)
    lineas = [f"{k}: {v}" for k, v in anot.items() if k != "Clips" and v != ""]
    leidos = [leer_frames(r) for r in rutas]
    vistas = [f for f, _ in leidos]
    fps = leidos[0][1]
    total = max(len(v) for v in vistas)

    i, pausa = 0, False
    while True:
        img = mosaico(vistas, i)
        texto(img, lineas, i, total)
        cv2.imshow(f"accion {args.accion}", img)
        tecla = cv2.waitKey(0 if pausa else int(1000 / fps)) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord(" "):
            pausa = not pausa
        elif tecla == ord("r"):
            i = 0
        elif tecla == ord(",") and pausa:
            i = max(i - 1, 0)
        elif tecla == ord(".") and pausa:
            i = min(i + 1, total - 1)
        elif not pausa:
            i = (i + 1) % total
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
