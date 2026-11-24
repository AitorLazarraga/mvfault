import json

import cv2
import numpy as np
import pytest

from vlm.keyframes import extraer_keyframes, indices_uniformes, indices_ventana, keyframes_multivista
from vlm.modelo import Descriptor, Simulado
from vlm.prompt import construir_prompt
from vlm import run_baseline


@pytest.fixture
def video(tmp_path):
    """Clip sintético de 125 frames (5 s a 25 fps) con el nº de frame dibujado."""
    ruta = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 720))
    for i in range(125):
        img = np.full((720, 1280, 3), i * 2 % 255, np.uint8)
        cv2.putText(img, str(i), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 8)
        w.write(img)
    w.release()
    return ruta


# --- keyframes ---------------------------------------------------------------

def test_indices_uniformes():
    idx = indices_uniformes(125, 4)
    assert len(idx) == 4 and idx == sorted(idx) and 0 <= idx[0] and idx[-1] < 125


def test_indices_ventana_incluye_centro_y_recorta():
    idx = indices_ventana(125, 6, centro=110, antes=20, despues=50)
    assert 110 in idx and max(idx) <= 124 and min(idx) >= 90


def test_extraer_keyframes_redimensiona(video):
    kfs = extraer_keyframes(video, n=5)
    assert len(kfs) == 5
    assert max(kfs[0].imagen.size) == 640
    assert kfs[0].vista == "clip.mp4"


def test_multivista_respeta_maximo(video):
    kfs = keyframes_multivista([video, video, video], n_por_vista=4, max_total=8)
    assert len(kfs) == 8


# --- prompt ------------------------------------------------------------------

def test_prompt_contiene_bloques():
    p = construir_prompt(7, ["clip.mp4"], [4])
    assert "Regla 12" in p and "jugada_id: 7" in p and "```json" in p
    assert "Datos de la escena" not in p


def test_prompt_con_datos_escena():
    p = construir_prompt(7, ["clip.mp4"], [4], datos_escena='{"t":0}')
    assert "Datos de la escena" in p


# --- descriptor -----------------------------------------------------------------

def test_descriptor_fija_jugada_id(video):
    r = Descriptor(Simulado()).describir(42, extraer_keyframes(video, n=2), "prompt")
    assert r.ok and r.descripcion.jugada_id == 42 and len(r.salidas_crudas) == 1


def test_descriptor_reintenta_tras_json_invalido(video):
    valido = "```json\n" + json.dumps(Simulado.RESPUESTA) + "\n```"
    backend = Simulado(["No sé qué pasa.", valido])
    r = Descriptor(backend, reintentos=1).describir(1, extraer_keyframes(video, n=2), "p")
    assert r.ok and len(r.salidas_crudas) == 2
    # el segundo mensaje incluye la respuesta anterior y la corrección
    assert [m["role"] for m in backend.llamadas[1]] == ["user", "assistant", "user"]


def test_descriptor_falla_tras_agotar_reintentos(video):
    malo = '```json\n{"hay_infraccion": true}\n```'
    r = Descriptor(Simulado([malo, malo]), reintentos=1).describir(
        1, extraer_keyframes(video, n=2), "p")
    assert not r.ok and r.error and len(r.salidas_crudas) == 2


# --- pipeline completo -----------------------------------------------------------

def test_run_baseline_simulado_y_reanudacion(video, tmp_path):
    salida = tmp_path / "res.jsonl"
    args = ["--video", str(video), "--id", "3", "--simulado", "--salida", str(salida),
            "--guardar-frames", str(tmp_path / "frames")]
    assert run_baseline.main(args) == 0
    filas = [json.loads(l) for l in salida.read_text(encoding="utf-8").splitlines()]
    assert len(filas) == 1 and filas[0]["ok"]
    assert filas[0]["decision"]["tarjeta"] == "ninguna"
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 4

    # segunda ejecución: la jugada ya está hecha, no se duplica
    run_baseline.main(args)
    assert len(salida.read_text(encoding="utf-8").splitlines()) == 1


def test_run_baseline_manifiesto(video, tmp_path):
    manifiesto = tmp_path / "jugadas.jsonl"
    manifiesto.write_text(
        json.dumps({"jugada_id": 1, "videos": [str(video)], "centros": [75]}) + "\n"
        + json.dumps({"jugada_id": 2, "videos": [str(tmp_path / "no_existe.mp4")]}) + "\n",
        encoding="utf-8")
    salida = tmp_path / "res.jsonl"
    run_baseline.main(["--manifest", str(manifiesto), "--simulado", "--salida", str(salida)])
    filas = {f["jugada_id"]: f for f in map(json.loads, salida.read_text().splitlines())}
    assert filas[1]["ok"] and 75 in filas[1]["frames"]
    assert not filas[2]["ok"]  # vídeo inexistente: error registrado, no se para
