"""
Etapa 1 — baseline sin entrenar.

    vídeo(s) → keyframes → prompt → VLM → DescripcionJugada → motor de reglas → JSONL

Uso con un vídeo suelto (prueba rápida):
    python -m vlm.run_baseline --video falta.mp4 --id 1 --salida resultados/prueba.jsonl

Con varias vistas de la misma jugada:
    python -m vlm.run_baseline --video vista0.mp4 --video vista1.mp4 --id 1

Con un manifiesto (una jugada por línea):
    python -m vlm.run_baseline --manifest jugadas.jsonl --salida resultados/baseline.jsonl
    # cada línea: {"jugada_id": 1, "videos": ["a.mp4", "b.mp4"], "centros": [75, null]}

Sin GPU, para comprobar el circuito:
    python -m vlm.run_baseline --video falta.mp4 --id 1 --simulado

Si el fichero de salida ya existe, se saltan las jugadas ya procesadas
(se puede cortar y reanudar una ejecución larga).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from rules.engine import decidir
from vlm.keyframes import Keyframe, keyframes_multivista
from vlm.modelo import MODELO_POR_DEFECTO, Descriptor, QwenVL, Simulado
from vlm.prompt import construir_prompt


def leer_manifiesto(ruta: Path) -> Iterator[Dict]:
    with open(ruta, encoding="utf-8") as f:
        for n, linea in enumerate(f, 1):
            linea = linea.strip()
            if not linea:
                continue
            j = json.loads(linea)
            if "jugada_id" not in j or "videos" not in j:
                raise ValueError(f"{ruta}:{n}: faltan 'jugada_id' o 'videos'")
            yield j


def ya_procesadas(salida: Path) -> set:
    if not salida.exists():
        return set()
    ids = set()
    with open(salida, encoding="utf-8") as f:
        for linea in f:
            try:
                ids.add(json.loads(linea)["jugada_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def guardar_frames(keyframes: List[Keyframe], carpeta: Path, jugada_id: int) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    for kf in keyframes:
        kf.imagen.save(carpeta / f"{jugada_id}_{Path(kf.vista).stem}_f{kf.indice:04d}.jpg")


def procesar_jugada(jugada: Dict, descriptor: Descriptor, args) -> Dict:
    jid = int(jugada["jugada_id"])
    videos = [Path(v) for v in jugada["videos"]]
    centros: Optional[List[Optional[int]]] = jugada.get("centros")

    keyframes = keyframes_multivista(videos, n_por_vista=args.n_por_vista,
                                     max_total=args.max_imagenes, centros=centros)
    if not keyframes:
        return {"jugada_id": jid, "ok": False, "error": "No se pudo leer ningún frame"}
    if args.guardar_frames:
        guardar_frames(keyframes, Path(args.guardar_frames), jid)

    vistas = list(dict.fromkeys(kf.vista for kf in keyframes))
    por_vista = [sum(kf.vista == v for kf in keyframes) for v in vistas]
    prompt = construir_prompt(jid, vistas, por_vista, datos_escena=jugada.get("datos_escena"))

    r = descriptor.describir(jid, keyframes, prompt)
    fila = {
        "jugada_id": jid,
        "ok": r.ok,
        "vistas": vistas,
        "frames": [kf.indice for kf in keyframes],
        "segundos": round(r.segundos, 2),
        "intentos": len(r.salidas_crudas),
        "salidas_crudas": r.salidas_crudas,
        "error": r.error,
    }
    if r.ok:
        fila["descripcion"] = r.descripcion.model_dump(mode="json")
        fila["decision"] = decidir(r.descripcion).model_dump(mode="json")
    return fila


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MVFault — baseline Etapa 1")
    fuente = p.add_mutually_exclusive_group(required=True)
    fuente.add_argument("--video", action="append", help="Vídeo de la jugada (repetible)")
    fuente.add_argument("--manifest", type=Path, help="JSONL con una jugada por línea")
    p.add_argument("--id", type=int, default=0, help="jugada_id si se usa --video")
    p.add_argument("--salida", type=Path, default=Path("resultados/baseline.jsonl"))
    p.add_argument("--modelo", default=MODELO_POR_DEFECTO)
    p.add_argument("--sin-cuantizar", action="store_true", help="Cargar en fp16 (más VRAM)")
    p.add_argument("--simulado", action="store_true", help="Sin modelo: probar el circuito")
    p.add_argument("--n-por-vista", type=int, default=4)
    p.add_argument("--max-imagenes", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--reintentos", type=int, default=1)
    p.add_argument("--guardar-frames", help="Carpeta donde guardar los keyframes usados")
    args = p.parse_args(argv)

    if args.video:
        jugadas = [{"jugada_id": args.id, "videos": args.video}]
    else:
        jugadas = list(leer_manifiesto(args.manifest))

    hechas = ya_procesadas(args.salida)
    pendientes = [j for j in jugadas if int(j["jugada_id"]) not in hechas]
    print(f"{len(jugadas)} jugadas, {len(hechas)} ya procesadas, {len(pendientes)} pendientes")
    if not pendientes:
        return 0

    backend = Simulado() if args.simulado else QwenVL(
        args.modelo, cuantizar_4bit=not args.sin_cuantizar, max_tokens=args.max_tokens)
    descriptor = Descriptor(backend, reintentos=args.reintentos)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(args.salida, "a", encoding="utf-8") as f:
        for k, jugada in enumerate(pendientes, 1):
            try:
                fila = procesar_jugada(jugada, descriptor, args)
            except Exception as e:  # una jugada rota no para la ejecución
                fila = {"jugada_id": jugada.get("jugada_id"), "ok": False,
                        "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            f.flush()
            n_ok += fila["ok"]
            dec = fila.get("decision", {})
            print(f"[{k}/{len(pendientes)}] jugada {fila['jugada_id']}: "
                  + (f"{dec.get('tarjeta')} · {dec.get('motivo')}" if fila["ok"]
                     else f"ERROR {fila.get('error')}"))

    print(f"Terminado: {n_ok}/{len(pendientes)} válidas → {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
