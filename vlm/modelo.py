"""
Envoltorio del VLM: genera la descripción de la jugada y la valida con el esquema.

    Descriptor  -> lógica común: construir mensajes, validar, reintentar.
    Backend     -> quién genera el texto:
        - QwenVL:    modelo real con transformers (opcionalmente en 4 bits).
        - Simulado:  respuesta fija, para probar el pipeline sin GPU.

El jugada_id lo pone siempre el pipeline, nunca el modelo.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from pydantic import ValidationError

from scene.schema import DescripcionJugada, extraer_json
from vlm.keyframes import Keyframe
from vlm.prompt import prompt_correccion

MODELO_POR_DEFECTO = "Qwen/Qwen3-VL-4B-Instruct"

Mensajes = List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class Backend(Protocol):
    def generar(self, mensajes: Mensajes) -> str: ...


class QwenVL:
    """VLM de Hugging Face (probado pensando en la familia Qwen3-VL)."""

    def __init__(self, modelo: str = MODELO_POR_DEFECTO, cuantizar_4bit: bool = True,
                 max_tokens: int = 1024):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        kwargs: Dict[str, Any] = {"device_map": "auto"}
        if cuantizar_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            kwargs["dtype"] = torch.float16

        self.processor = AutoProcessor.from_pretrained(modelo)
        self.model = AutoModelForImageTextToText.from_pretrained(modelo, **kwargs)
        self.model.eval()
        self.max_tokens = max_tokens
        self._torch = torch

    def generar(self, mensajes: Mensajes) -> str:
        entradas = self.processor.apply_chat_template(
            mensajes,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self._torch.inference_mode():
            salida = self.model.generate(**entradas, max_new_tokens=self.max_tokens,
                                         do_sample=False)
        nuevos = salida[:, entradas["input_ids"].shape[1]:]
        return self.processor.batch_decode(nuevos, skip_special_tokens=True)[0]


class Simulado:
    """Backend sin modelo: devuelve siempre la misma descripción válida."""

    RESPUESTA = {
        "jugada_id": 0,
        "dorsal_infractor": None,
        "hay_infraccion": True,
        "infraccion_ifab": "zancadillear",
        "nivel": "imprudente",
        "balon_en_juego": True,
        "victima": "rival",
        "parte_infractor": "pie_interior",
        "zona_impacto": "tobillo_pie",
        "balon_disputable": True,
        "intento_jugar_balon": True,
        "en_area": False,
        "ataque_prometedor": False,
        "ocasion_gol": {"direccion_porteria": None, "control_balon": None,
                        "cerca_area_penal": None, "defensores_entre": None},
        "ventaja_aplicada": False,
        "descripcion_ifab": "Respuesta simulada: zancadilla imprudente en disputa del balón",
    }

    def __init__(self, respuestas: Optional[Sequence[str]] = None):
        self._respuestas = list(respuestas) if respuestas else None
        self.llamadas: List[Mensajes] = []

    def generar(self, mensajes: Mensajes) -> str:
        self.llamadas.append(mensajes)
        if self._respuestas:
            return self._respuestas.pop(0)
        return "Análisis: respuesta simulada.\n```json\n" + \
            json.dumps(self.RESPUESTA, ensure_ascii=False) + "\n```"


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------

@dataclass
class ResultadoVLM:
    jugada_id: int
    descripcion: Optional[DescripcionJugada]
    salidas_crudas: List[str] = field(default_factory=list)
    error: Optional[str] = None
    segundos: float = 0.0

    @property
    def ok(self) -> bool:
        return self.descripcion is not None


def _mensaje_usuario(keyframes: Sequence[Keyframe], texto: str) -> Dict[str, Any]:
    contenido: List[Dict[str, Any]] = [{"type": "image", "image": kf.imagen} for kf in keyframes]
    contenido.append({"type": "text", "text": texto})
    return {"role": "user", "content": contenido}


def _validar(texto: str, jugada_id: int) -> DescripcionJugada:
    datos = json.loads(extraer_json(texto))
    if not isinstance(datos, dict):
        raise ValueError("El JSON no es un objeto.")
    datos["jugada_id"] = jugada_id  # lo fija el pipeline, no el modelo
    return DescripcionJugada.model_validate(datos)


class Descriptor:
    def __init__(self, backend: Backend, reintentos: int = 1):
        self.backend = backend
        self.reintentos = reintentos

    def describir(self, jugada_id: int, keyframes: Sequence[Keyframe],
                  prompt: str) -> ResultadoVLM:
        t0 = time.perf_counter()
        mensajes: Mensajes = [_mensaje_usuario(keyframes, prompt)]
        resultado = ResultadoVLM(jugada_id=jugada_id, descripcion=None)

        for intento in range(self.reintentos + 1):
            texto = self.backend.generar(mensajes)
            resultado.salidas_crudas.append(texto)
            try:
                resultado.descripcion = _validar(texto, jugada_id)
                resultado.error = None
                break
            except (ValidationError, ValueError, json.JSONDecodeError) as e:
                resultado.error = f"{type(e).__name__}: {e}"
                mensajes = mensajes + [
                    {"role": "assistant", "content": [{"type": "text", "text": texto}]},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_correccion(str(e))}]},
                ]

        resultado.segundos = time.perf_counter() - t0
        return resultado
