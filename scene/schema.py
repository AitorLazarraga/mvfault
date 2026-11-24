"""
Esquema de la descripción de una jugada con contacto (MVFault).

El VLM genera este JSON y el motor de reglas (rules/) lo lee para decidir
la sanción según la Regla 12 de la IFAB. Todos los campos que el motor
usa son cerrados (enum / bool) para que las reglas no dependan de texto libre.

Convención: None = "desconocido / no aplica" (p. ej. no se ven las líneas
del área en el plano), nunca se usa False para decir "no lo sé".
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


# ---------------------------------------------------------------------------
# Enumeraciones
# ---------------------------------------------------------------------------

class InfraccionIFAB(str, Enum):
    """Infracciones de la Regla 12 que implican contacto (+ simulación)."""

    # Graduables: se valoran como imprudente / temeraria / fuerza excesiva
    CARGAR = "cargar"
    SALTAR_SOBRE = "saltar_sobre"
    DAR_PATADA = "dar_patada"
    EMPUJAR = "empujar"
    CODAZO = "codazo"
    CABEZAZO = "cabezazo"
    ENTRADA_O_DISPUTA = "entrada_o_disputa"
    ZANCADILLEAR = "zancadillear"

    # Siempre con intención: agresión / juego brusco grave (roja)
    GOLPEAR = "golpear"

    # No graduables: tiro libre directo sin valorar la fuerza
    SUJETAR = "sujetar"
    IMPEDIR_AVANCE_CON_CONTACTO = "impedir_avance_con_contacto"

    # Infracción del atacante
    SIMULACION = "simulacion"


INFRACCIONES_GRADUABLES = {
    InfraccionIFAB.CARGAR,
    InfraccionIFAB.SALTAR_SOBRE,
    InfraccionIFAB.DAR_PATADA,
    InfraccionIFAB.EMPUJAR,
    InfraccionIFAB.CODAZO,
    InfraccionIFAB.CABEZAZO,
    InfraccionIFAB.ENTRADA_O_DISPUTA,
    InfraccionIFAB.ZANCADILLEAR,
}


INFRACCIONES_SIEMPRE_ROJA = {
    InfraccionIFAB.GOLPEAR,
}

# Parte del cuerpo que exige cada infracción (si se informa parte_infractor)
PARTE_REQUERIDA = {
    InfraccionIFAB.CODAZO: "codo",
    InfraccionIFAB.CABEZAZO: "cabeza",
}


class Nivel(str, Enum):
    """Valoración de la fuerza según la Regla 12."""

    IMPRUDENTE = "imprudente"            # tiro libre, sin tarjeta
    TEMERARIA = "temeraria"              # amarilla
    FUERZA_EXCESIVA = "fuerza_excesiva"  # roja


class ParteInfractor(str, Enum):
    """Parte del cuerpo con la que el infractor hace el contacto."""

    PIE_PLANTA = "pie_planta"            # tacos por delante
    PIE_EMPEINE = "pie_empeine"
    PIE_INTERIOR = "pie_interior"
    PIERNA = "pierna"
    RODILLA = "rodilla"
    CADERA_CUERPO = "cadera_cuerpo"
    BRAZO = "brazo"
    CODO = "codo"
    MANO = "mano"
    CABEZA = "cabeza"
    OTRO = "otro"


class ZonaImpacto(str, Enum):
    """Zona del rival donde se produce el contacto."""

    TOBILLO_PIE = "tobillo_pie"
    ESPINILLA = "espinilla"
    RODILLA = "rodilla"
    MUSLO = "muslo"
    CADERA = "cadera"
    TRONCO = "tronco"
    ESPALDA = "espalda"
    BRAZO = "brazo"
    CUELLO_CABEZA = "cuello_cabeza"
    OTRO = "otro"


class Victima(str, Enum):
    """A quién afecta la acción. Solo 'rival' permite tiro libre por contacto."""

    RIVAL = "rival"
    COMPANERO = "companero"
    ARBITRO = "arbitro"              # cualquier miembro del equipo arbitral
    OFICIAL_EQUIPO = "oficial_equipo"
    ESPECTADOR = "espectador"
    OTRO = "otro"


def extraer_json(texto: str) -> str:
    """Devuelve el texto del objeto JSON de una respuesta del VLM.

    Acepta bloques ```json ... ``` o un objeto JSON suelto entre texto.
    Lanza ValueError si no encuentra ninguno.
    """
    bloque = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto, re.DOTALL)
    if bloque:
        return bloque.group(1)
    inicio, fin = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fin == -1 or fin < inicio:
        raise ValueError("No se encontró ningún objeto JSON en la salida.")
    return texto[inicio : fin + 1]


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

class OcasionGol(BaseModel):
    """Criterios de ocasión manifiesta de gol (OMG).

    El motor de reglas decide si hay OMG combinando estos criterios;
    el VLM solo los observa.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    direccion_porteria: Optional[bool] = Field(
        None, description="El atacante avanzaba en dirección general a la portería."
    )
    control_balon: Optional[bool] = Field(
        None, description="El atacante tenía o iba a obtener el control del balón."
    )
    cerca_area_penal: Optional[bool] = Field(
        None,
        description="Las líneas del área penal están próximas a la jugada. "
        "None si no se ven en el plano.",
    )
    defensores_entre: Optional[int] = Field(
        None, ge=0, le=11,
        description="Defensores (incluido el portero) entre la jugada y la portería.",
    )


class DescripcionJugada(BaseModel):
    """Descripción estructurada de una jugada con contacto."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    jugada_id: int = Field(..., ge=0)
    dorsal_infractor: Optional[int] = Field(
        None, ge=1, le=99,
        description="Dorsal de quien comete la infracción (en simulación, el atacante).",
    )

    # --- Infracción -------------------------------------------------------
    hay_infraccion: bool
    infraccion_ifab: Optional[InfraccionIFAB] = None
    nivel: Optional[Nivel] = Field(
        None,
        description="Obligatorio si la infracción es graduable; None si no hay infracción.",
    )

    # --- Contexto de la acción ---------------------------------------------
    balon_en_juego: Optional[bool] = Field(
        None, description="El balón estaba en juego en el momento de la acción."
    )
    victima: Victima = Field(
        Victima.RIVAL, description="A quién se dirige la acción."
    )

    # --- Contacto ---------------------------------------------------------
    parte_infractor: Optional[ParteInfractor] = None
    zona_impacto: Optional[ZonaImpacto] = None
    balon_disputable: Optional[bool] = Field(
        None, description="El balón estaba a distancia de juego para el infractor."
    )
    intento_jugar_balon: Optional[bool] = Field(
        None, description="El infractor intentó jugar el balón (clave en el área)."
    )

    # --- Situación de juego -----------------------------------------------
    en_area: Optional[bool] = Field(
        None, description="La infracción ocurre dentro del área penal del infractor."
    )
    ataque_prometedor: Optional[bool] = Field(
        None, description="La infracción corta un ataque prometedor."
    )
    ocasion_gol: OcasionGol = Field(default_factory=OcasionGol)
    ventaja_aplicada: Optional[bool] = Field(
        None, description="El árbitro deja seguir el juego aplicando ventaja."
    )

    # --- Justificación (no la usa el motor) -------------------------------
    descripcion_ifab: str = Field(..., min_length=10, max_length=400)
    confianza: Optional[float] = Field(None, ge=0.0, le=1.0)

    # --- Coherencia entre campos ------------------------------------------
    @model_validator(mode="after")
    def _coherencia(self) -> "DescripcionJugada":
        if not self.hay_infraccion:
            if self.infraccion_ifab is not None or self.nivel is not None:
                raise ValueError(
                    "Sin infracción, 'infraccion_ifab' y 'nivel' deben ser null."
                )
            return self

        if self.infraccion_ifab is None:
            raise ValueError("Con infracción, 'infraccion_ifab' es obligatorio.")

        infraccion = InfraccionIFAB(self.infraccion_ifab)

        if infraccion in INFRACCIONES_GRADUABLES and self.nivel is None:
            raise ValueError(
                f"'{self.infraccion_ifab}' es graduable: 'nivel' es obligatorio."
            )

        if infraccion in INFRACCIONES_SIEMPRE_ROJA and self.nivel not in (
            None, Nivel.FUERZA_EXCESIVA.value
        ):
            raise ValueError(
                f"'{self.infraccion_ifab}' implica intención: 'nivel' solo puede "
                "ser null o 'fuerza_excesiva'."
            )

        parte = PARTE_REQUERIDA.get(infraccion)
        if parte and self.parte_infractor is not None and self.parte_infractor != parte:
            raise ValueError(
                f"'{self.infraccion_ifab}' exige parte_infractor='{parte}'."
            )
        return self

    # --- Utilidades para el VLM -------------------------------------------
    @classmethod
    def schema_para_prompt(cls) -> str:
        """JSON Schema compacto para meterlo en el prompt del VLM."""
        return json.dumps(cls.model_json_schema(), ensure_ascii=False)

    @classmethod
    def desde_salida_llm(cls, texto: str) -> "DescripcionJugada":
        """Extrae y valida el JSON de la respuesta del VLM.

        Acepta bloques ```json ... ``` o texto con un objeto JSON suelto.
        Lanza ValidationError o ValueError si no es válido.
        """
        return cls.model_validate_json(extraer_json(texto))


__all__ = [
    "extraer_json",
    "InfraccionIFAB",
    "INFRACCIONES_GRADUABLES",
    "INFRACCIONES_SIEMPRE_ROJA",
    "PARTE_REQUERIDA",
    "Nivel",
    "ParteInfractor",
    "ZonaImpacto",
    "Victima",
    "OcasionGol",
    "DescripcionJugada",
    "ValidationError",
]