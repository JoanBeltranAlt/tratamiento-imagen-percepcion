# -*- coding: utf-8 -*-
"""Diagnóstico de las cinco capturas y tratamiento de la imagen problemática.

Mide lo que pide el enunciado —resolución, canales, brillo, contraste, nitidez,
formato y tamaño— y deja los resultados en JSON para que el informe los cite
sin transcribirlos a mano.
"""
import json
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
IMGS = AQUI / "imagenes"
SALIDA = AQUI / "resultados"
SALIDA.mkdir(exist_ok=True)

# el orden importa: la primera es la de referencia
CONDICIONES = [
    ("01_referencia", "Referencia"),
    ("02_baja_luz", "Baja iluminación"),
    ("03_desenfoque", "Desenfoque o movimiento"),
    ("04_fondo_complejo", "Fondo complejo"),
    ("05_angulo", "Cambio de ángulo o distancia"),
]


def medir(ruta):
    """Devuelve las características que pide el diagnóstico."""
    img = cv2.imread(str(ruta))
    if img is None:
        raise ValueError(f"No se pudo leer {ruta}")
    alto, ancho = img.shape[:2]
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # brillo: media de intensidad sobre 255
    brillo = float(gris.mean())
    # contraste: desviación estándar; a mayor dispersión, más contraste
    contraste = float(gris.std())
    # nitidez: varianza del laplaciano, el detector de bordes de segundo orden
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    # saturación media en HSV, útil para ver el efecto de la poca luz
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturacion = float(hsv[:, :, 1].mean())

    # recorrido real del histograma: qué parte del rango 0-255 se usa
    p1, p99 = np.percentile(gris, [1, 99])
    # píxeles quemados o aplastados contra los extremos
    subexp = float((gris < 30).mean() * 100)
    sobrexp = float((gris > 225).mean() * 100)

    # entropía: cuánta información lleva el histograma, en bits
    hist = cv2.calcHist([gris], [0], None, [256], [0, 256]).ravel()
    p = hist / hist.sum()
    p = p[p > 0]
    entropia = float(-(p * np.log2(p)).sum())

    return dict(
        archivo=ruta.name,
        formato=ruta.suffix.lstrip(".").upper(),
        peso_kb=round(ruta.stat().st_size / 1024, 1),
        ancho=int(ancho), alto=int(alto),
        resolucion=f"{ancho}x{alto}",
        megapixeles=round(ancho * alto / 1e6, 2),
        canales=int(img.shape[2]) if img.ndim == 3 else 1,
        profundidad=str(img.dtype),
        brillo=round(brillo, 1),
        contraste=round(contraste, 1),
        nitidez=round(nitidez, 1),
        saturacion=round(saturacion, 1),
        entropia=round(entropia, 2),
        tenengrad=round(tenengrad(gris), 1),
        densidad_bordes=round(densidad_bordes(gris), 2),
        ruido_zona_plana=round(ruido_zona_plana(gris), 2),
        rango_util=f"{int(p1)}–{int(p99)}",
        amplitud_rango=int(p99 - p1),
        pct_subexpuesto=round(subexp, 1),
        pct_sobreexpuesto=round(sobrexp, 1),
    )


def main():
    datos = {"capturas": [], "versiones": {"opencv": cv2.__version__,
                                           "numpy": np.__version__}}
    for base, descripcion in CONDICIONES:
        encontrada = next((p for p in sorted(IMGS.glob(base + ".*"))), None)
        if encontrada is None:
            print(f"  falta: {base}.*")
            continue
        m = medir(encontrada)
        m["condicion"] = descripcion
        datos["capturas"].append(m)

    if not datos["capturas"]:
        print("No hay imágenes en", IMGS)
        return

    (AQUI / "diagnostico.json").write_text(
        json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")

    cols = ["condicion", "resolucion", "canales", "brillo", "contraste",
            "nitidez", "entropia", "peso_kb", "formato"]
    anchos = [max(len(c), max(len(str(f[c])) for f in datos["capturas"]))
              for c in cols]
    print("  ".join(c.upper().ljust(w) for c, w in zip(cols, anchos)))
    for f in datos["capturas"]:
        print("  ".join(str(f[c]).ljust(w) for c, w in zip(cols, anchos)))




# ════════════════ Tratamiento de la imagen problemática ═════════════════
def tratar_baja_luz(img):
    """Realce de una imagen subexpuesta.

    Se trabaja sobre el canal L de LAB para no alterar el color: CLAHE
    redistribuye la luminancia por zonas, en vez de estirar el histograma
    global como haría equalizeHist, que aquí quemaría las partes ya claras.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    tratada = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    # el realce amplifica el ruido del sensor, que en poca luz es alto
    return cv2.fastNlMeansDenoisingColored(tratada, None, 6, 6, 7, 21)


def tratar_desenfoque(img):
    """Realce de bordes por máscara de desenfoque (unsharp masking).

    Se resta una versión suavizada de la original, lo que acentúa las
    transiciones. No recupera información perdida: solo aumenta el contraste
    local en los bordes que todavía son distinguibles.
    """
    suave = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.6, suave, -0.6, 0)


def tenengrad(gris):
    """Energía de gradiente de Sobel: segunda medida de nitidez.

    Se reporta junto al laplaciano porque la máscara de enfoque sube ambos
    aunque no recupere detalle real, y conviene verlo con dos indicadores."""
    gx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    return float((gx ** 2 + gy ** 2).mean())


def ruido_zona_plana(gris, lado=64):
    """Desviación típica del parche más uniforme de la imagen.

    Realzar brillo o bordes amplifica el ruido del sensor. Medirlo en una
    zona sin detalle permite comprobar si el tratamiento lo disparó."""
    h, w = gris.shape
    mejor = None
    for y in range(0, h - lado, max(1, (h - lado) // 8)):
        for x in range(0, w - lado, max(1, (w - lado) // 8)):
            d = float(gris[y:y + lado, x:x + lado].std())
            if mejor is None or d < mejor:
                mejor = d
    return mejor if mejor is not None else float(gris.std())


def densidad_bordes(gris):
    """Porcentaje de píxeles que Canny marca como borde."""
    return float((cv2.Canny(cv2.GaussianBlur(gris, (5, 5), 0),
                            80, 160) == 255).mean() * 100)


def comparar(antes, despues):
    """Métricas antes y después, más el cambio relativo de cada una."""
    def perfil(x):
        g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
        p = hist / hist.sum(); p = p[p > 0]
        p1, p99 = np.percentile(g, [1, 99])
        return dict(brillo=round(float(g.mean()), 1),
                    contraste=round(float(g.std()), 1),
                    nitidez=round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1),
                    tenengrad=round(tenengrad(g), 1),
                    entropia=round(float(-(p * np.log2(p)).sum()), 2),
                    amplitud_rango=int(p99 - p1),
                    pct_subexpuesto=round(float((g < 30).mean() * 100), 1),
                    pct_sobreexpuesto=round(float((g > 225).mean() * 100), 1),
                    densidad_bordes=round(densidad_bordes(g), 2),
                    ruido_zona_plana=round(ruido_zona_plana(g), 2))
    a, d = perfil(antes), perfil(despues)
    cambio = {k: (round(100 * (d[k] - a[k]) / a[k], 1) if a[k] else None)
              for k in a}
    return {"antes": a, "despues": d, "cambio_pct": cambio}


if __name__ == "__main__":
    main()
