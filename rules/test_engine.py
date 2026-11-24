from rules.engine import decidir, es_ocasion_manifiesta
from scene.schema import DescripcionJugada


def jugada(**cambios):
    datos = {
        "jugada_id": 1,
        "dorsal_infractor": 4,
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
        "ocasion_gol": {
            "direccion_porteria": False,
            "control_balon": False,
            "cerca_area_penal": False,
            "defensores_entre": 5,
        },
        "ventaja_aplicada": False,
        "descripcion_ifab": "Descripción de prueba de la jugada",
    }
    og = cambios.pop("ocasion_gol", None)
    datos.update(cambios)
    if og:
        datos["ocasion_gol"].update(og)
    return DescripcionJugada.model_validate(datos)


OMG = {"direccion_porteria": True, "control_balon": True,
       "cerca_area_penal": True, "defensores_entre": 1}


# --- 1. Sin infracción ------------------------------------------------------

def test_sin_infraccion():
    d = decidir(jugada(hay_infraccion=False, infraccion_ifab=None, nivel=None))
    assert not d.falta and d.reanudacion == "continua" and d.tarjeta == "ninguna"


# --- 2. Simulación ------------------------------------------------------------

def test_simulacion():
    d = decidir(jugada(infraccion_ifab="simulacion", nivel=None, dorsal_infractor=9))
    assert d.tarjeta == "amarilla" and d.reanudacion == "tiro_libre_indirecto"
    assert d.dorsal == 9


# --- 3. Víctima no rival / balón fuera ---------------------------------------

def test_golpe_con_balon_fuera_de_juego():
    d = decidir(jugada(infraccion_ifab="golpear", nivel=None, parte_infractor="mano",
                       balon_en_juego=False))
    assert d.tarjeta == "roja" and "Conducta violenta" in d.motivo
    assert d.reanudacion == "segun_interrupcion"


def test_golpe_a_espectador_balon_en_juego():
    d = decidir(jugada(infraccion_ifab="golpear", nivel=None, parte_infractor="mano",
                       victima="espectador"))
    assert d.tarjeta == "roja" and d.reanudacion == "balon_a_tierra"


def test_empujon_a_companero():
    d = decidir(jugada(infraccion_ifab="empujar", nivel="fuerza_excesiva",
                       victima="companero"))
    assert d.tarjeta == "roja" and d.reanudacion == "tiro_libre_directo"


def test_no_rival_salta_capa_omg():
    d = decidir(jugada(infraccion_ifab="empujar", nivel="imprudente",
                       victima="arbitro", ocasion_gol=OMG))
    assert d.tarjeta == "ninguna"


# --- 4. Sujetar / impedir ------------------------------------------------------

def test_agarron_sin_ataque():
    d = decidir(jugada(infraccion_ifab="sujetar", nivel=None))
    assert d.tarjeta == "ninguna" and d.reanudacion == "tiro_libre_directo"


def test_agarron_corta_ataque_prometedor():
    d = decidir(jugada(infraccion_ifab="sujetar", nivel=None, ataque_prometedor=True))
    assert d.tarjeta == "amarilla" and "ataque prometedor" in d.motivo


def test_agarron_omg_en_area_es_roja():
    d = decidir(jugada(infraccion_ifab="sujetar", nivel=None, en_area=True,
                       intento_jugar_balon=False, ocasion_gol=OMG))
    assert d.tarjeta == "roja" and d.reanudacion == "penalti"


# --- 5. Graduables y golpear -----------------------------------------------------

def test_imprudente():
    d = decidir(jugada(nivel="imprudente"))
    assert d.tarjeta == "ninguna" and d.reanudacion == "tiro_libre_directo"


def test_temeraria():
    d = decidir(jugada(nivel="temeraria"))
    assert d.tarjeta == "amarilla"


def test_fuerza_excesiva_con_disputa_jbg():
    d = decidir(jugada(nivel="fuerza_excesiva"))
    assert d.tarjeta == "roja" and d.motivo.startswith("Juego brusco grave")


def test_fuerza_excesiva_sin_disputa_cv():
    d = decidir(jugada(infraccion_ifab="empujar", nivel="fuerza_excesiva",
                       balon_disputable=False))
    assert d.tarjeta == "roja" and d.motivo.startswith("Conducta violenta")


def test_golpear_sin_balon():
    d = decidir(jugada(infraccion_ifab="golpear", nivel=None, parte_infractor="mano",
                       balon_disputable=False))
    assert d.tarjeta == "roja" and d.motivo == "Conducta violenta"


def test_imprudente_en_area_penalti():
    d = decidir(jugada(en_area=True))
    assert d.reanudacion == "penalti"


# --- 6. OMG / ataque prometedor / ventaja ------------------------------------------

def test_omg_fuera_area_roja():
    d = decidir(jugada(ocasion_gol=OMG))
    assert d.tarjeta == "roja" and "ocasión manifiesta" in d.motivo


def test_omg_en_area_con_intento_amarilla():
    d = decidir(jugada(en_area=True, ocasion_gol=OMG))
    assert d.tarjeta == "amarilla" and d.reanudacion == "penalti"


def test_omg_con_ventaja_baja_un_nivel():
    d = decidir(jugada(ocasion_gol=OMG, ventaja_aplicada=True))
    assert d.tarjeta == "amarilla" and d.reanudacion == "ventaja"


def test_ataque_prometedor_con_ventaja_sin_tarjeta():
    d = decidir(jugada(ataque_prometedor=True, ventaja_aplicada=True))
    assert d.tarjeta == "ninguna"


def test_ataque_prometedor_en_area_con_intento_sin_tarjeta():
    d = decidir(jugada(ataque_prometedor=True, en_area=True))
    assert d.tarjeta == "ninguna" and d.reanudacion == "penalti"


def test_gana_la_tarjeta_mas_alta():
    # Entrada con fuerza excesiva en OMG dentro del área con intento:
    # la OMG daría amarilla, pero la fuerza excesiva manda roja.
    d = decidir(jugada(nivel="fuerza_excesiva", en_area=True, ocasion_gol=OMG))
    assert d.tarjeta == "roja" and d.motivo.startswith("Juego brusco grave")


def test_fuerza_excesiva_con_ventaja_sigue_roja():
    d = decidir(jugada(nivel="fuerza_excesiva", ventaja_aplicada=True))
    assert d.tarjeta == "roja"


# --- Criterios de OMG ------------------------------------------------------------

def test_omg_control_por_disputa():
    j = jugada(ocasion_gol={**OMG, "control_balon": False})
    assert es_ocasion_manifiesta(j)  # balón disputable + intento


def test_omg_sin_control_ni_disputa():
    j = jugada(balon_disputable=False, ocasion_gol={**OMG, "control_balon": False})
    assert not es_ocasion_manifiesta(j)


def test_omg_demasiados_defensores():
    assert not es_ocasion_manifiesta(jugada(ocasion_gol={**OMG, "defensores_entre": 3}))


def test_omg_dos_defensores_si():
    assert es_ocasion_manifiesta(jugada(ocasion_gol={**OMG, "defensores_entre": 2}))


def test_omg_criterio_null_no_es_manifiesta():
    assert not es_ocasion_manifiesta(jugada(ocasion_gol={**OMG, "cerca_area_penal": None}))


# --- Avisos ----------------------------------------------------------------------

def test_aviso_en_area_desconocido():
    d = decidir(jugada(en_area=None))
    assert d.reanudacion == "tiro_libre_directo" and d.avisos


# --- Ajustes tras revisar el texto IFAB 2026/27 ------------------------------------

def test_omg_en_area_empujon_es_roja_aunque_haya_intento():
    # "agarrar, tirar, empujar... el infractor deberá ser expulsado"
    d = decidir(jugada(infraccion_ifab="empujar", nivel="imprudente", en_area=True,
                       ocasion_gol=OMG))
    assert d.tarjeta == "roja"


def test_omg_en_area_balon_no_disputable_es_roja():
    # "imposibilidad de jugar el balón" -> expulsión
    d = decidir(jugada(en_area=True, balon_disputable=False,
                       ocasion_gol={**OMG, "control_balon": True}))
    assert d.tarjeta == "roja"


def test_ataque_prometedor_en_area_agarron_es_amarilla():
    d = decidir(jugada(infraccion_ifab="sujetar", nivel=None, en_area=True,
                       ataque_prometedor=True))
    assert d.tarjeta == "amarilla" and d.reanudacion == "penalti"


def test_aviso_ventaja_en_juego_brusco_grave():
    d = decidir(jugada(nivel="fuerza_excesiva", ventaja_aplicada=True))
    assert d.tarjeta == "roja" and any("ventaja" in a for a in d.avisos)
