"""
Construcción del prompt para el VLM.

El prompt tiene cuatro bloques:
    1. Rol y tarea.
    2. Reglas IFAB (texto literal de rules/ifab_extracto.md).
    3. Definiciones de los campos del esquema + JSON Schema.
    4. Datos de la jugada: vistas/frames y, desde la Etapa 2, el NDJSON de pose.

La salida pedida es: un análisis breve y después el JSON en un bloque ```json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

from scene.schema import DescripcionJugada

RAIZ = Path(__file__).resolve().parents[1]
RUTA_REGLAS = RAIZ / "rules" / "ifab_extracto.md"


ROL = """Eres un árbitro de fútbol experto que revisa jugadas con contacto en vídeo.
Tu tarea NO es decidir la sanción: es DESCRIBIR la jugada con precisión en el
formato JSON indicado. Un motor de reglas aplicará después la Regla 12 de la IFAB
a tu descripción para decidir la sanción.

Reglas de trabajo:
- Describe solo lo que se ve. Si un dato no se puede saber con las imágenes, usa null.
  null significa "no se puede saber"; false significa "se ve que no".
- Las imágenes son fotogramas de una o varias cámaras de la MISMA jugada, en orden
  temporal dentro de cada cámara. Las cámaras no están sincronizadas entre sí.
- Usa exactamente los valores permitidos en los campos cerrados."""


DEFINICIONES = """Definiciones de los campos (además del JSON Schema):

infraccion_ifab:
- cargar: choque con el cuerpo (hombro, cadera) contra el adversario.
- saltar_sobre: saltar encima o contra el adversario.
- dar_patada: patada o intento de patada al adversario.
- empujar: empujar con manos, brazos o cuerpo.
- codazo: contacto con el codo, típico en saltos o disputas; puede ser accidental.
- cabezazo: choque o contacto con la cabeza; puede ser accidental en una disputa.
- golpear: golpe deliberado al adversario (puñetazo, manotazo, golpe con el brazo
  sin disputar el balón). Implica intención.
- entrada_o_disputa: entrada o disputa del balón con contacto sobre el adversario.
- zancadillear: poner la zancadilla o intentarlo.
- sujetar: agarrar o retener al adversario con manos, brazos o cuerpo.
- impedir_avance_con_contacto: obstaculizar al adversario con contacto cuando el
  balón no está a distancia de juego.
- simulacion: el atacante finge haber recibido una falta. El infractor es el atacante.

nivel (solo para infracciones con contacto contra un adversario):
- imprudente: falta de atención o precaución al disputar el balón.
- temeraria: sin tener en cuenta el riesgo o las consecuencias para el adversario.
- fuerza_excesiva: se excede en la fuerza o pone en peligro la integridad física.
- Para sujetar, impedir_avance_con_contacto y simulacion puede ser null.
- Para golpear solo puede ser null o fuerza_excesiva.

Otros:
- dorsal_infractor: dorsal de quien comete la infracción si se lee con claridad; si no, null.
- balon_disputable: el balón estaba a distancia de juego del infractor.
- intento_jugar_balon: el infractor intentó jugar o disputar el balón.
- en_area: la infracción ocurre dentro del área penal del equipo del infractor.
- ataque_prometedor: la infracción corta un ataque prometedor del adversario.
- ocasion_gol.cerca_area_penal: las líneas del área penal se ven próximas a la jugada.
- ocasion_gol.defensores_entre: defensores (incluido el portero) entre la jugada y la portería.
- ventaja_aplicada: el juego continúa tras la falta porque el árbitro concede ventaja.
- descripcion_ifab: una frase con la terminología de la IFAB, p. ej.
  "Zancadillea al rival de manera temeraria en disputa del balón"."""


FORMATO_SALIDA = """Responde en dos partes:
1. "Análisis:" con un máximo de 5 frases describiendo la jugada.
2. El JSON en un bloque ```json ... ``` que cumpla el esquema. No añadas campos."""


def cargar_reglas(ruta: Path = RUTA_REGLAS) -> str:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encuentra {ruta}. Es el extracto de la IFAB que necesita el prompt."
        )
    return ruta.read_text(encoding="utf-8")


def describir_vistas(vistas: Sequence[str], frames_por_vista: Sequence[int]) -> str:
    lineas = ["Imágenes adjuntas, en este orden:"]
    for v, n in zip(vistas, frames_por_vista):
        lineas.append(f"- cámara '{v}': {n} fotogramas")
    return "\n".join(lineas)


def construir_prompt(
    jugada_id: int,
    vistas: Sequence[str],
    frames_por_vista: Sequence[int],
    datos_escena: Optional[str] = None,
    reglas: Optional[str] = None,
) -> str:
    """Texto completo del prompt. Las imágenes se pasan aparte al modelo.

    `datos_escena` es el NDJSON (o su resumen) de la Etapa 2; en la Etapa 1 es None.
    """
    bloques: List[str] = [
        ROL,
        "## Reglas de Juego IFAB (extracto)\n" + (reglas or cargar_reglas()),
        DEFINICIONES,
        "## JSON Schema de la salida\n" + json.dumps(
            DescripcionJugada.model_json_schema(), ensure_ascii=False),
        f"## Jugada\njugada_id: {jugada_id}\n" + describir_vistas(vistas, frames_por_vista),
    ]
    if datos_escena:
        bloques.append(
            "## Datos de la escena (pose y tracking, NDJSON)\n"
            "Datos medidos automáticamente; pueden tener errores si hay oclusiones.\n"
            + datos_escena
        )
    bloques.append(FORMATO_SALIDA)
    return "\n\n".join(bloques)


def prompt_correccion(error: str) -> str:
    """Mensaje para el reintento cuando el JSON no valida."""
    return (
        "Tu JSON anterior no es válido según el esquema. Error:\n"
        f"{error}\n\n"
        "Corrige el JSON y devuélvelo completo en un bloque ```json ... ```, sin análisis."
    )
