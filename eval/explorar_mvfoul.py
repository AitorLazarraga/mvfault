"""
Explora SoccerNet-MVFoul: campos de anotación, valores y formato de los vídeos.

Uso:
    python eval/explorar_mvfoul.py data/mvfouls
    python eval/explorar_mvfoul.py data/mvfouls --videos 20   # inspecciona 20 clips

Genera:
    - Resumen por pantalla: cuántas acciones hay por split, qué campos tiene cada
      acción y la distribución de valores de cada campo (y de los campos de clip).
    - data/mvfouls/acciones.csv: una fila por acción, para abrirla en una hoja.

No asume la estructura exacta: busca todos los annotations.json bajo la ruta.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

IGNORAR_CLIP = {"Url", "Timestamp"}


def cargar_anotaciones(raiz: Path):
    """Devuelve {split: {id_accion: dict}} a partir de cada annotations.json."""
    splits = {}
    for ruta in sorted(raiz.rglob("annotations.json")):
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
        split = ruta.parent.name
        if isinstance(datos, dict) and "Actions" in datos:
            splits[split] = datos["Actions"]
        else:
            print(f"[!] {ruta}: estructura inesperada, claves: {list(datos)[:10]}")
    return splits


def resumen_campos(splits):
    valores = defaultdict(Counter)
    valores_clip = defaultdict(Counter)
    clips_por_accion = Counter()

    for acciones in splits.values():
        for accion in acciones.values():
            for campo, valor in accion.items():
                if campo == "Clips":
                    clips_por_accion[len(valor)] += 1
                    for clip in valor:
                        for c, v in clip.items():
                            if c not in IGNORAR_CLIP:
                                valores_clip[c][str(v)] += 1
                else:
                    valores[campo][str(valor) if valor != "" else "<vacío>"] += 1
    return valores, valores_clip, clips_por_accion


def imprimir_counter(titulo, contador, max_items=25):
    print(f"\n  {titulo}")
    total = sum(contador.values())
    for valor, n in contador.most_common(max_items):
        print(f"    {valor:<40} {n:>6}  ({100 * n / total:5.1f} %)")
    if len(contador) > max_items:
        print(f"    ... y {len(contador) - max_items} valores más")


def exportar_csv(splits, destino: Path):
    campos = sorted({c for acc in splits.values() for a in acc.values() for c in a
                     if c != "Clips"})
    with open(destino, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "accion", *campos, "n_clips"])
        for split, acciones in splits.items():
            for id_accion, a in acciones.items():
                w.writerow([split, id_accion, *[a.get(c, "") for c in campos],
                            len(a.get("Clips", []))])
    print(f"\nCSV guardado en {destino}")


def inspeccionar_videos(raiz: Path, n: int):
    try:
        import cv2
    except ImportError:
        print("\n[!] Instala opencv-python para inspeccionar los vídeos.")
        return
    videos = sorted(raiz.rglob("*.mp4"))[:n]
    if not videos:
        print("\n[!] No se encontraron .mp4 (¿están descomprimidos?)")
        return
    print(f"\nVídeos ({len(videos)} de muestra):")
    formatos = Counter()
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        formatos[f"{w}x{h} @ {fps:.0f} fps, {frames} frames ({frames / fps if fps else 0:.1f} s)"] += 1
    imprimir_counter("Formato de los clips", formatos)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raiz", type=Path)
    p.add_argument("--videos", type=int, default=10)
    args = p.parse_args()

    splits = cargar_anotaciones(args.raiz)
    if not splits:
        print("No se encontró ningún annotations.json")
        return

    print("Acciones por split:")
    for split, acciones in splits.items():
        print(f"  {split:<12} {len(acciones):>5}")

    valores, valores_clip, clips_por_accion = resumen_campos(splits)
    print("\n=== Campos de cada acción ===")
    for campo, contador in valores.items():
        imprimir_counter(campo, contador)

    print("\n=== Campos de cada clip ===")
    for campo, contador in valores_clip.items():
        imprimir_counter(campo, contador)
    imprimir_counter("Clips por acción", Counter({str(k): v for k, v in clips_por_accion.items()}))

    exportar_csv(splits, args.raiz / "acciones.csv")
    inspeccionar_videos(args.raiz, args.videos)


if __name__ == "__main__":
    main()
