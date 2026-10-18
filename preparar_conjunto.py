# -*- coding: utf-8 -*-
"""Arma el conjunto de cinco condiciones a partir del dataset LOL.

Las condiciones 1 y 2 son capturas reales: el mismo encuadre fotografiado con
exposición normal y con poca luz, tomado del dataset LOL (Wei et al., 2018),
que se construyó variando ISO y tiempo de exposición sobre la misma escena.

Las condiciones 3, 4 y 5 se derivan por software de la imagen de referencia,
porque el dataset no las incluye. El informe lo declara de forma explícita: una
degradación sintética reproduce el efecto sobre las métricas, pero no equivale
a una captura real, y eso condiciona la interpretación.
"""
import shutil
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
DESTINO = AQUI / "imagenes"
DESTINO.mkdir(exist_ok=True)
rng = np.random.default_rng(7)


def desenfoque_movimiento(img, largo=21, angulo=18.0):
    """Estela de movimiento: convolución con un kernel lineal.

    Se usa en vez de GaussianBlur porque el barrido de una cámara movida es
    direccional, no isotrópico, y eso se nota en el espectro de la imagen.
    """
    k = np.zeros((largo, largo), np.float32)
    k[largo // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((largo / 2 - 0.5, largo / 2 - 0.5), angulo, 1)
    k = cv2.warpAffine(k, M, (largo, largo))
    k /= k.sum()
    return cv2.filter2D(img, -1, k)


def fondo_complejo(img, recortes):
    """Superpone recortes de otras escenas alrededor del objeto central.

    La zona central se conserva intacta para que el objeto siga siendo el
    mismo; lo que cambia es el entorno que lo rodea.
    """
    h, w = img.shape[:2]
    lienzo = img.copy()
    # el objeto ocupa la franja central superior; solo se protege esa zona
    px1, px2 = int(w * 0.34), int(w * 0.70)
    py1, py2 = int(h * 0.02), int(h * 0.55)
    colocados = 0
    intentos = 0
    while colocados < 9 and intentos < 400:
        intentos += 1
        r = recortes[colocados % len(recortes)]
        rh = int(rng.integers(h // 6, h // 3))
        rw = int(rng.integers(w // 6, w // 3))
        parche = cv2.resize(r, (rw, rh))
        x = int(rng.integers(0, max(1, w - rw)))
        y = int(rng.integers(0, max(1, h - rh)))
        if x + rw > px1 and x < px2 and y + rh > py1 and y < py2:
            continue        # pisaría el objeto
        lienzo[y:y + rh, x:x + rw] = parche
        colocados += 1
    return lienzo


def otro_angulo(img, grados=-16.0, escala=0.74):
    """Rota y aleja la escena, como una toma desde otra posición."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), grados, escala)
    borde = tuple(int(v) for v in img[2:12, 2:12].reshape(-1, 3).mean(0))
    return cv2.warpAffine(img, M, (w, h), borderValue=borde)


def construir(ruta_normal, ruta_baja, rutas_relleno):
    ref = cv2.imread(str(ruta_normal))
    baja = cv2.imread(str(ruta_baja))
    if ref is None or baja is None:
        raise ValueError("No se pudieron leer las imágenes del dataset")

    cv2.imwrite(str(DESTINO / "01_referencia.png"), ref)
    cv2.imwrite(str(DESTINO / "02_baja_luz.png"), baja)
    cv2.imwrite(str(DESTINO / "03_desenfoque.png"),
                desenfoque_movimiento(ref))
    recortes = [cv2.imread(str(p)) for p in rutas_relleno]
    recortes = [r for r in recortes if r is not None]
    cv2.imwrite(str(DESTINO / "04_fondo_complejo.png"),
                fondo_complejo(ref, recortes))
    cv2.imwrite(str(DESTINO / "05_angulo.png"), otro_angulo(ref))

    for p in sorted(DESTINO.iterdir()):
        im = cv2.imread(str(p))
        print(f"  {p.name:24s} {im.shape[1]}x{im.shape[0]}  "
              f"{p.stat().st_size / 1024:.0f} KB")
