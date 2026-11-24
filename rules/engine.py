"""
Motor de reglas de MVFault (Regla 12 IFAB, jugadas con contacto).

Recibe una DescripcionJugada (salida del VLM) y devuelve la decisión arbitral:
    {falta, dorsal, reanudacion, tarjeta, motivo}

Orden de evaluación:
    1. Sin infracción
    2. Simulación
    3. Víctima que no es rival o balón fuera de juego (conducta violenta)
    4. Sujetar / impedir el avance con contacto
    5. Infracciones graduables y golpear
    6. Capa ocasión manifiesta de gol / ataque prometedor (+ ventaja)
       -> se queda la tarjeta más alta entre 4-5 y 6

Convención: un criterio a None ("no se puede saber") nunca da ocasión
manifiesta: si no se ve con claridad, no es manifiesta.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from scene.schema import (
    INFRACCIONES_GRADUABLES,
    INFRACCIONES_SIEMPRE_ROJA,
    DescripcionJugada,
    InfraccionIFAB,
    Nivel,
    Victima,
)

MAX_DEFENSORES_OMG = 2  # hasta un defensa y el portero


# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------

class Tarjeta(str, Enum):
    NINGUNA = "ninguna"
    AMARILLA = "amarilla"
    ROJA = "roja"


_RANGO = {Tarjeta.NINGUNA: 0, Tarjeta.AMARILLA: 1, Tarjeta.ROJA: 2}
_POR_RANGO = {v: k for k, v in _RANGO.items()}


class Reanudacion(str, Enum):
    CONTINUA = "continua"                        # sin infracción
    VENTAJA = "ventaja"                          # se deja seguir; tarjeta al parar
    TIRO_LIBRE_DIRECTO = "tiro_libre_directo"
    TIRO_LIBRE_INDIRECTO = "tiro_libre_indirecto"
    PENALTI = "penalti"
    SEGUN_INTERRUPCION = "segun_interrupcion"    # balón ya fuera de juego
    BALON_A_TIERRA = "balon_a_tierra"


class Decision(BaseModel):
    falta: bool
    dorsal: Optional[int] = None
    reanudacion: Reanudacion
    tarjeta: Tarjeta
    motivo: str
    avisos: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _enum(valor, tipo):
    """Los modelos guardan valores (use_enum_values); los pasamos a enum."""
    return None if valor is None else tipo(valor)


def hay_disputa(d: DescripcionJugada) -> bool:
    """Disputa del balón: balón disputable y no se descarta el intento de jugarlo."""
    return d.balon_disputable is True and d.intento_jugar_balon is not False


def es_ocasion_manifiesta(d: DescripcionJugada) -> bool:
    og = d.ocasion_gol
    criterios = (og.direccion_porteria, og.cerca_area_penal, og.control_balon,
                 og.defensores_entre)
    if any(c is None for c in criterios):
        return False
    control = og.control_balon or hay_disputa(d)
    return (
        og.direccion_porteria
        and og.cerca_area_penal
        and control
        and og.defensores_entre <= MAX_DEFENSORES_OMG
    )


# Infracciones que nunca cuentan como "intento de jugar o disputar el balón"
# (Regla 12: "agarrar, tirar, empujar, imposibilidad de jugar el balón...")
NO_CUENTAN_COMO_INTENTO = {
    InfraccionIFAB.SUJETAR,
    InfraccionIFAB.EMPUJAR,
    InfraccionIFAB.IMPEDIR_AVANCE_CON_CONTACTO,
}


def intento_reglamentario(d: DescripcionJugada, infr: InfraccionIFAB) -> bool:
    """La infracción se cometió al intentar jugar o disputar el balón."""
    return hay_disputa(d) and d.intento_jugar_balon is True \
        and infr not in NO_CUENTAN_COMO_INTENTO


def _reducir(t: Tarjeta) -> Tarjeta:
    return _POR_RANGO[max(_RANGO[t] - 1, 0)]


def _reanudacion_falta(d: DescripcionJugada, avisos: List[str]) -> Reanudacion:
    if d.ventaja_aplicada:
        return Reanudacion.VENTAJA
    if d.en_area is None:
        avisos.append("en_area desconocido: se asume tiro libre directo.")
    return Reanudacion.PENALTI if d.en_area else Reanudacion.TIRO_LIBRE_DIRECTO


def _motivo_roja(d: DescripcionJugada) -> str:
    return "Juego brusco grave" if hay_disputa(d) else "Conducta violenta"


# ---------------------------------------------------------------------------
# Pasos
# ---------------------------------------------------------------------------

def _paso_no_rival_o_balon_fuera(d: DescripcionJugada, infr: InfraccionIFAB,
                                 nivel: Optional[Nivel]) -> Decision:
    victima = Victima(d.victima)

    if nivel == Nivel.IMPRUDENTE:
        tarjeta, motivo = Tarjeta.NINGUNA, "Contacto imprudente"
    elif nivel == Nivel.TEMERARIA and infr not in INFRACCIONES_SIEMPRE_ROJA:
        tarjeta, motivo = Tarjeta.AMARILLA, "Conducta antideportiva"
    else:
        tarjeta, motivo = Tarjeta.ROJA, "Conducta violenta"

    if d.balon_en_juego is False:
        reanudacion = Reanudacion.SEGUN_INTERRUPCION
    elif victima in (Victima.ESPECTADOR, Victima.OTRO):
        reanudacion = Reanudacion.BALON_A_TIERRA
    else:  # compañero, árbitro, oficial: tiro libre directo / penalti
        reanudacion = Reanudacion.PENALTI if d.en_area else Reanudacion.TIRO_LIBRE_DIRECTO

    return Decision(falta=True, dorsal=d.dorsal_infractor, reanudacion=reanudacion,
                    tarjeta=tarjeta, motivo=f"{motivo} ({victima.value})")


def _tarjeta_por_infraccion(d: DescripcionJugada, infr: InfraccionIFAB,
                            nivel: Optional[Nivel]) -> Tuple[Tarjeta, str]:
    if infr in (InfraccionIFAB.SUJETAR, InfraccionIFAB.IMPEDIR_AVANCE_CON_CONTACTO):
        return Tarjeta.NINGUNA, f"Tiro libre directo: {infr.value}"

    if infr in INFRACCIONES_SIEMPRE_ROJA:
        return Tarjeta.ROJA, _motivo_roja(d)

    # Graduables
    if nivel == Nivel.IMPRUDENTE:
        return Tarjeta.NINGUNA, f"{infr.value} de forma imprudente"
    if nivel == Nivel.TEMERARIA:
        return Tarjeta.AMARILLA, f"Conducta antideportiva: {infr.value} de forma temeraria"
    return Tarjeta.ROJA, f"{_motivo_roja(d)}: {infr.value} con fuerza excesiva"


def _tarjeta_por_situacion(d: DescripcionJugada,
                           infr: InfraccionIFAB) -> Tuple[Tarjeta, Optional[str]]:
    intento = intento_reglamentario(d, infr)

    if es_ocasion_manifiesta(d):
        if d.en_area and intento:
            tarjeta = Tarjeta.AMARILLA
        else:
            tarjeta = Tarjeta.ROJA
        motivo = "Evitar ocasión manifiesta de gol"
    elif d.ataque_prometedor:
        tarjeta = Tarjeta.NINGUNA if (d.en_area and intento) else Tarjeta.AMARILLA
        motivo = "Cortar ataque prometedor"
    else:
        return Tarjeta.NINGUNA, None

    if d.ventaja_aplicada:
        tarjeta = _reducir(tarjeta)
        motivo += " (ventaja aplicada)"
    return tarjeta, motivo


# ---------------------------------------------------------------------------
# Entrada principal
# ---------------------------------------------------------------------------

def decidir(d: DescripcionJugada) -> Decision:
    avisos: List[str] = []
    infr = _enum(d.infraccion_ifab, InfraccionIFAB)
    nivel = _enum(d.nivel, Nivel)

    # 1. Sin infracción
    if not d.hay_infraccion:
        return Decision(falta=False, reanudacion=Reanudacion.CONTINUA,
                        tarjeta=Tarjeta.NINGUNA, motivo="Sin infracción")

    # 2. Simulación
    if infr == InfraccionIFAB.SIMULACION:
        return Decision(falta=True, dorsal=d.dorsal_infractor,
                        reanudacion=Reanudacion.TIRO_LIBRE_INDIRECTO,
                        tarjeta=Tarjeta.AMARILLA,
                        motivo="Conducta antideportiva: simulación")

    # 3. Víctima no rival o balón fuera de juego
    if Victima(d.victima) != Victima.RIVAL or d.balon_en_juego is False:
        return _paso_no_rival_o_balon_fuera(d, infr, nivel)

    if d.balon_en_juego is None:
        avisos.append("balon_en_juego desconocido: se asume en juego.")

    # 4-5. Tarjeta por la propia infracción
    t_infr, m_infr = _tarjeta_por_infraccion(d, infr, nivel)

    # 6. Capa OMG / ataque prometedor
    t_sit, m_sit = _tarjeta_por_situacion(d, infr)

    if (d.ventaja_aplicada and t_infr == Tarjeta.ROJA
            and not es_ocasion_manifiesta(d)):
        avisos.append("Regla 12: no se debe conceder ventaja en juego brusco grave "
                      "o conducta violenta salvo ocasión manifiesta de gol.")

    if m_sit and _RANGO[t_sit] > _RANGO[t_infr]:
        tarjeta, motivo = t_sit, m_sit
    else:
        tarjeta, motivo = t_infr, m_infr

    return Decision(falta=True, dorsal=d.dorsal_infractor,
                    reanudacion=_reanudacion_falta(d, avisos),
                    tarjeta=tarjeta, motivo=motivo, avisos=avisos)


__all__ = ["Decision", "Reanudacion", "Tarjeta", "decidir",
           "es_ocasion_manifiesta", "hay_disputa", "intento_reglamentario",
           "MAX_DEFENSORES_OMG"]
