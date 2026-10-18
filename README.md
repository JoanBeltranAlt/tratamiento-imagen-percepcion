# Tratamiento de imagen · Percepción Computacional

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green)
![NumPy](https://img.shields.io/badge/NumPy-2.5-orange)
![Colab](https://img.shields.io/badge/Google%20Colab-ready-yellow)

Práctica de las **Semanas 8 y 9** del CADI Percepción Computacional
(experiencia IMAGEN). Diagnostica cinco condiciones de captura sobre la misma
escena, identifica la imagen inservible y le aplica un tratamiento que responde
al problema medido.

**Repositorio:** https://github.com/JoanBeltranAlt/tratamiento-imagen-percepcion
**Entorno:** Google Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoanBeltranAlt/tratamiento-imagen-percepcion/blob/main/Tratamiento_Imagen_Semana8_9.ipynb)

---

## Origen de las imágenes

Las condiciones **1 y 2 son capturas fotográficas reales** del mismo encuadre,
tomadas del conjunto [LOL](https://daooshee.github.io/BMVC2018website/)
(Wei et al., 2018), construido variando el tiempo de exposición y la ISO de la
cámara sobre escenas reales.

Las condiciones **3, 4 y 5 se derivan por software** de la imagen de
referencia, porque el conjunto no las incluye. El código que las genera está en
`preparar_conjunto.py` y en el notebook. Una degradación sintética reproduce el
efecto sobre las métricas, pero no todo lo que ocurre en una captura real, y el
informe lo declara.

## Diagnóstico

| Condición | Brillo | Contraste | Nitidez | Entropía | <30 (%) |
|---|---:|---:|---:|---:|---:|
| 1 · Referencia | 158,5 | 51,1 | 180,3 | 7,38 | 1,2 |
| 2 · Baja iluminación | **23,5** | **17,1** | 45,9 | 5,57 | **80,8** |
| 3 · Desenfoque | 158,5 | 49,3 | **6,8** | 7,39 | 0,9 |
| 4 · Fondo complejo | 151,4 | 50,7 | *1.118,4* | 7,39 | 0,4 |
| 5 · Cambio de ángulo | 168,2 | 39,1 | 139,6 | 4,98 | 0,6 |

Dos lecturas que vale la pena destacar:

- La **baja iluminación** es la única degradación que impide el procesamiento
  posterior: cuatro de cada cinco píxeles quedan por debajo del valor 30 y la
  densidad de bordes cae de 1,89 % a 0,04 %.
- El **fondo complejo** obtiene la nitidez más alta del conjunto pese a ser la
  imagen más difícil de interpretar. La varianza del laplaciano mide variación
  local, no calidad, y cuenta el desorden del fondo como si fuera detalle útil.

## Tratamiento

CLAHE sobre el canal L del espacio LAB, seguido de reducción de ruido por
medias no locales. Se descartaron la ecualización global (amplifica el ruido de
las sombras) y la corrección gamma (aclara pero no separa tonos vecinos).

| Métrica | Antes | Después | Cambio | Referencia |
|---|---:|---:|---:|---:|
| Brillo | 23,5 | 42,8 | +82,1 % | 158,5 |
| Contraste | 17,1 | 29,2 | +70,8 % | 51,1 |
| Píxeles bajo 30 (%) | 80,8 | 28,7 | −64,5 % | 1,2 |
| Densidad de bordes (%) | 0,04 | 0,52 | +1.200 % | 1,89 |
| Ruido en zona plana | 1,10 | 0,66 | −40,0 % | 1,69 |

El ruido **baja** en lugar de subir porque el filtro de medias no locales
suprime más del que CLAHE introduce. Verificarlo importa: sin esa comprobación,
un aumento de nitidez podría estar reflejando ruido amplificado.

La mejora no equivale a una recuperación. CLAHE redistribuye la información que
la imagen contiene; no inventa la que el sensor nunca capturó.

## Sobre las métricas de evaluación

PSNR y SSIM son métricas *full-reference*: comparan dos imágenes píxel a píxel
y exigen que estén alineadas. Aquí no existe esa pareja, porque la imagen de
referencia es otra toma con su propio encuadre. Usarlas mediría la diferencia
entre capturas, no el efecto del tratamiento. Por eso la evaluación compara las
mismas métricas antes y después, contrastándolas con el rango de la referencia.

## Contenido

| Archivo | Descripción |
|---|---|
| `Tratamiento_Imagen_Semana8_9.ipynb` | Notebook con la práctica completa |
| `diagnostico.py` | Medición de las características y tratamiento |
| `preparar_conjunto.py` | Generación de las condiciones 3, 4 y 5 |
| `imagenes/` | Las cinco condiciones de captura |
| `resultados/` | Imagen tratada y figuras comparativas |
| `diagnostico.json`, `tratamiento.json` | Mediciones que cita el informe |

## Referencias

- Wei, C., Wang, W., Yang, W., & Liu, J. (2018). *Deep Retinex decomposition
  for low-light enhancement* [Ponencia]. British Machine Vision Conference.
  https://arxiv.org/abs/1808.04560
- Pizer, S. M., et al. (1987). Adaptive histogram equalization and its
  variations. *Computer Vision, Graphics, and Image Processing, 39*(3),
  355–368. https://doi.org/10.1016/S0734-189X(87)80186-X
- Canny, J. (1986). A computational approach to edge detection. *IEEE TPAMI,
  PAMI-8*(6), 679–698. https://doi.org/10.1109/TPAMI.1986.4767851
- OpenCV. (2026). *OpenCV documentation* (Versión 5.0.0).
  https://docs.opencv.org/5.x/

---

Joan Schneider Beltrán Delgado · Universidad de Cundinamarca
