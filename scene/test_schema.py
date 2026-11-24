import pytest

from scene.schema import DescripcionJugada, ValidationError


def base(**cambios):
    datos = {
        "jugada_id": 123,
        "hay_infraccion": True,
        "infraccion_ifab": "zancadillear",
        "nivel": "temeraria",
        "parte_infractor": "pie_planta",
        "zona_impacto": "tobillo_pie",
        "balon_disputable": True,
        "intento_jugar_balon": True,
        "en_area": False,
        "ataque_prometedor": False,
        "ocasion_gol": {
            "direccion_porteria": False,
            "control_balon": False,
            "cerca_area_penal": None,
            "defensores_entre": 3,
        },
        "descripcion_ifab": "Zancadillea al rival de manera temeraria en disputa del balón",
    }
    datos.update(cambios)
    return datos


def test_jugada_valida():
    d = DescripcionJugada.model_validate(base())
    assert d.nivel == "temeraria"
    assert d.ocasion_gol.cerca_area_penal is None


def test_sin_infraccion_valida():
    d = DescripcionJugada.model_validate(
        base(hay_infraccion=False, infraccion_ifab=None, nivel=None,
             descripcion_ifab="Disputa legal del balón hombro con hombro")
    )
    assert not d.hay_infraccion


def test_sin_infraccion_con_nivel_falla():
    with pytest.raises(ValidationError):
        DescripcionJugada.model_validate(base(hay_infraccion=False, infraccion_ifab=None))


def test_infraccion_sin_tipo_falla():
    with pytest.raises(ValidationError):
        DescripcionJugada.model_validate(base(infraccion_ifab=None))


def test_graduable_sin_nivel_falla():
    with pytest.raises(ValidationError):
        DescripcionJugada.model_validate(base(nivel=None))


def test_no_graduable_sin_nivel_valida():
    d = DescripcionJugada.model_validate(base(infraccion_ifab="sujetar", nivel=None))
    assert d.nivel is None


def test_enum_invalido_falla():
    with pytest.raises(ValidationError):
        DescripcionJugada.model_validate(base(nivel="muy_fuerte"))


def test_campo_extra_falla():
    with pytest.raises(ValidationError):
        DescripcionJugada.model_validate(base(minuto=34))


def test_desde_salida_llm_con_bloque():
    import json
    texto = "Análisis:\n```json\n" + json.dumps(base()) + "\n```\nFin."
    d = DescripcionJugada.desde_salida_llm(texto)
    assert d.jugada_id == 123


def test_desde_salida_llm_sin_json_falla():
    with pytest.raises(ValueError):
        DescripcionJugada.desde_salida_llm("No puedo determinarlo.")


def test_schema_para_prompt():
    s = DescripcionJugada.schema_para_prompt()
    assert "zancadillear" in s and "fuerza_excesiva" in s


def test_codazo_graduable():
    with pytest.raises(ValidationError):
        DescripcionJugada.model_validate(base(infraccion_ifab="codazo", nivel=None,
                                              parte_infractor="codo"))
    d = DescripcionJugada.model_validate(base(infraccion_ifab="codazo", nivel="imprudente",
                                              parte_infractor="codo"))
    assert d.nivel == "imprudente"


def test_codazo_con_pie_falla():
    with pytest.raises(ValidationError):
        DescripcionJugada.model_validate(base(infraccion_ifab="codazo",
                                              parte_infractor="pie_planta"))


def test_cabezazo_sin_parte_valida():
    d = DescripcionJugada.model_validate(base(infraccion_ifab="cabezazo",
                                              parte_infractor=None))
    assert d.infraccion_ifab == "cabezazo"


def test_golpear_solo_roja():
    d = DescripcionJugada.model_validate(base(infraccion_ifab="golpear", nivel=None,
                                              parte_infractor="mano"))
    assert d.nivel is None
    d = DescripcionJugada.model_validate(base(infraccion_ifab="golpear",
                                              nivel="fuerza_excesiva",
                                              parte_infractor="mano"))
    assert d.nivel == "fuerza_excesiva"
    with pytest.raises(ValidationError):
        DescripcionJugada.model_validate(base(infraccion_ifab="golpear", nivel="temeraria",
                                              parte_infractor="mano"))


def test_simulacion():
    d = DescripcionJugada.model_validate(base(infraccion_ifab="simulacion", nivel=None,
                                              parte_infractor=None, zona_impacto=None))
    assert d.infraccion_ifab == "simulacion"


def test_victima_por_defecto_rival():
    d = DescripcionJugada.model_validate(base())
    assert d.victima == "rival" and d.balon_en_juego is None


def test_golpear_fuera_de_juego_a_espectador():
    d = DescripcionJugada.model_validate(base(infraccion_ifab="golpear", nivel=None,
                                              parte_infractor="mano",
                                              balon_en_juego=False,
                                              victima="espectador"))
    assert d.victima == "espectador"


def test_victima_invalida_falla():
    with pytest.raises(ValidationError):
        DescripcionJugada.model_validate(base(victima="publico"))
