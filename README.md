# Práctica 5: Detección Facial y Análisis Biométrico Avanzado

Este repositorio contiene la implementación de la **Práctica 5** de la asignatura de **Visión por Computador**. El proyecto integra pipelines de **detección facial**, **alineación con landmarks** y **clasificación de atributos biométricos** mediante **CNN**.

Incluye dos prototipos de **Realidad Aumentada (RA)** que procesan video en tiempo real aplicando transformaciones geométricas y superposición de gráficos basados en modelos de *Deep Learning*.

---

## Autor
[![GitHub](https://img.shields.io/badge/GitHub-Carlos%20Falcón-red?style=flat-square&logo=github)](https://github.com/carlosfc02)
---

## 🛠️ Stack Tecnológico

- **Computer Vision:** OpenCV (cv2), dlib (HOG + Linear SVM)  
- **Deep Learning:** TensorFlow/Keras (CNN)  
- **Imágenes:** NumPy, Pillow (GIFs)  
- **Audio:** pygame.mixer  

---

## 🚀 Prototipo 1: Clasificador de Emociones (CNN)

Pipeline de clasificación de expresiones faciales en tiempo real basado en un modelo entrenado con **FER-2013**.

### Arquitectura y Pipeline

1. **Detección Facial:** dlib (HOG + SVM)  
2. **Preprocesamiento (ROI):**
   - Extracción del rostro  
   - Escala de grises  
   - Redimensionamiento a **48×48×1**  
 
3. **Inferencia:** CNN Secuencial (Conv2D, MaxPooling, BatchNorm, Dropout) → Softmax (7 clases)

### Lógica de Realidad Aumentada

- **Felicidad** 😄: confeti generado con `cv2.circle` + assets estáticos  
- **Ira** 😡: superposición de GIF animado + renderizado de ojos rojos mediante detección de pupilas  

## Gif 

![Gif emociones](emotions.gif)
---


## 🐉 Prototipo 2: Transformación Interactiva (Geometría Facial)

Sistema RA basado en los **68 landmarks** del predictor de dlib (Kazemi & Sullivan).

### Lógica Algorítmica

#### Trigger Biométrico (Apertura Bucal)

- Se calcula la distancia vertical entre los landmarks **L62 (labio superior)** y **L66 (labio inferior)**.  


#### Renderizado de Assets

- Cálculo de centroides y escalas usando la distancia entre landmarks **0–16**.
- Ajuste dinámico de pelo y aura.

#### Efecto *Screen Shake*

- Transformación afín aleatoria por frame.  
- Aplicación con `cv2.warpAffine`.  
- Audio sincronizado con **pygame.mixer**.

## Gif 

![Gif ssj](ssj.gif)
---

---

## 🧠 Detalles de Implementación de Bajo Nivel

### 1. Alpha Blending Manual (`overlay_transparent`)
Debido a que OpenCV no maneja transparencia nativa, se implementa mezcla manual de canales RGBA/BGRA.

### 2. Decodificación de GIFs con Pillow
- Iteración de frames con `ImageSequence.Iterator`.  
- Conversión RGBA → BGRA para compatibilidad con OpenCV.  
- Almacenamiento en lista para reproducción en bucle.

---



