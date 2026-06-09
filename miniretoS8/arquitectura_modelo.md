# Arquitectura del Modelo YOLOv8 — miniretoS8

El modelo implementado es **YOLOv8** de Ultralytics, cargado desde `best.pt`
(o su versión optimizada `best.engine` para TensorRT en Jetson Nano).
Se utiliza para detección de señales de tráfico con 8 clases mapeadas a 6 tipos de señal.

---

## 1. Arquitectura de capas

YOLOv8 tiene tres bloques principales:

### Backbone — CSPDarknet

| Capa | Descripción |
|------|-------------|
| **Conv** (stem) | Conv2d 3×3, stride 2 → reduce la resolución a la mitad |
| **Conv** (downsample) | Conv2d 3×3, stride 2, se repite en cada etapa para reducir escala |
| **C2f** | Cross-Stage Partial con 2 bottlenecks: divide canales en dos ramas, una pasa por bloques residuales, ambas se concatenan |
| **SPPF** | Spatial Pyramid Pooling Fast: aplica MaxPool2d 5×5 tres veces en cascada (equivale a pools de 5, 9 y 13) y concatena los mapas → captura contexto multi-escala |

### Neck — PAN-FPN

| Capa | Descripción |
|------|-------------|
| **Upsample** | Interpolación bilineal ×2 para fusionar mapas de alta y baja resolución |
| **Concat** | Concatenación de features del backbone y el upsampling |
| **C2f** | Igual que en el backbone; refina los features fusionados |

### Head — Decoupled Detection

Tres cabezas independientes para escalas 80×80, 40×40 y 20×20 (relativas a la entrada):

- Rama de **regresión de bounding box** → predice `[x, y, w, h]`
- Rama de **clasificación** → predice probabilidad por clase

Las 8 clases del modelo se mapean a 6 señales mediante `_cls_to_sign_map`:

```python
_cls_to_sign_map = {
    7: 1,  # left
    2: 2,  # right
    0: 3,  # forward
    5: 4,  # stop
    1: 5,  # yield
    6: 6,  # roadwork
}
```

---

## 2. Funciones de activación

| Función | Dónde se aplica | Fórmula |
|---------|-----------------|---------|
| **SiLU** *(Sigmoid Linear Unit)* | Activación principal en **todas** las capas Conv, C2f y SPPF del backbone/neck | $f(x) = x \cdot \sigma(x) = \dfrac{x}{1+e^{-x}}$ |
| **Sigmoid** | Salida de clasificación del head | $\sigma(x) = \dfrac{1}{1+e^{-x}}$ |
| Sin activación | Salida de regresión de cajas | valor lineal |

**¿Por qué SiLU?**
Es suave, no satura completamente por debajo de cero (gradientes positivos para
$x < 0$), lo que mejora la convergencia frente a ReLU clásica. Es la función
de activación estándar de YOLOv5 en adelante.

---

## 3. Funcionamiento del kernel convolucional

Un kernel de tamaño $k \times k$ opera deslizándose sobre el mapa de entrada
y calculando:

$$\text{Salida}[i,j,c_{out}] = \sum_{c_{in}}\sum_{p=0}^{k-1}\sum_{q=0}^{k-1}
W[c_{in},p,q,c_{out}] \cdot \text{Entrada}[i \cdot s + p,\; j \cdot s + q,\; c_{in}]$$

donde $s$ es el **stride**.

### Kernels utilizados en YOLOv8

| Kernel | Stride | Padding | Efecto |
|--------|--------|---------|--------|
| **6×6** | 2 | 2 | Capa stem: captura features de bajo nivel con campo receptivo amplio |
| **3×3** | 1 | 1 | Extracción de features locales (bordes, texturas); mantiene dimensiones espaciales |
| **3×3** | 2 | 1 | Downsampling sin pooling: reduce ancho/alto a la mitad |
| **1×1** | 1 | 0 | Combinación lineal entre canales sin ver vecinos; comprime/expande canales en los bottlenecks de C2f con bajo costo computacional |
| **MaxPool 5×5** | 1 | 2 | Selecciona el valor máximo en ventana 5×5 (SPPF); aplicado 3 veces amplía el campo receptivo a 13×13 sin triplicar parámetros |

### Bloque estándar `Conv → BN → SiLU`

Después de cada convolución se aplica:

1. **BatchNorm** — normaliza la activación por mini-batch, estabiliza el entrenamiento.
2. **SiLU** — introduce no-linealidad.

Este bloque es el componente fundamental repetido en todo el backbone y el neck.

---

## 4. Métricas de evaluación obtenidas

Los valores provienen directamente del checkpoint `best.pt` (campo `train_metrics`),
correspondiente al mejor epoch alcanzado durante el entrenamiento en el dataset **SENAS2**
con 8 clases: `Forward`, `GiveWay`, `Right`, `Roundabout`, `Semaforo`, `Stop`,
`construction`, `left`.

### Configuración de entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Epochs configurados | 200 (detuvo en epoch 159, patience=100) |
| Tamaño de imagen | 640×640 px |
| Batch size | 16 |
| Optimizer | Auto (SGD) |
| lr₀ / lrf | 0.01 / 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3 |
| IoU threshold | 0.7 |
| Augmentación | Desactivada (augment=False) |

### Resultados en validación (mejor epoch)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Precision (B)** | **0.9842** (98.4%) | De cada detección positiva, el 98.4% es correcta |
| **Recall (B)** | **0.9514** (95.1%) | El modelo detecta el 95.1% de todas las señales presentes |
| **mAP@50 (B)** | **0.9875** (98.75%) | Área bajo la curva P-R con IoU≥0.50; excelente localización |
| **mAP@50-95 (B)** | **0.7577** (75.77%) | Promedio a IoU 0.50→0.95; refleja precisión estricta en bounding box |
| **Fitness** | **0.7807** | Métrica compuesta: `0.1·mAP50 + 0.9·mAP50-95` |
| val/box_loss | 0.874 | Error de regresión de bounding boxes (DFL) |
| val/cls_loss | 0.372 | Error de clasificación (BCE) |
| val/dfl_loss | 1.059 | Distribution Focal Loss (refinamiento de coordenadas) |

> **Nota:** La diferencia entre mAP@50 (98.75%) y mAP@50-95 (75.77%) indica que el modelo
> clasifica las señales con alta fiabilidad pero el ajuste fino del bounding box tiene
> margen de mejora, lo que es esperado dado que `augment=False` durante el entrenamiento.

---

## 5. Propuesta de métricas adicionales

Las métricas estándar de detección de objetos son necesarias pero insuficientes para
evaluar el desempeño **real del sistema en el PuzzleBot**. Se proponen las siguientes:

### 5.1 Métricas de inferencia en tiempo real (Jetson Nano)

| Métrica | Descripción | Cómo medirla |
|---------|-------------|--------------|
| **FPS (Frames por segundo)** | Throughput del pipeline completo: cámara → YOLO → publicación ROS | `time.perf_counter()` en el callback `_image_callback` |
| **Latencia end-to-end (ms)** | Tiempo desde captura de frame hasta publicación del mensaje | Timestamps ROS2 (`header.stamp`) comparados en el topic |
| **Tiempo de inferencia GPU (ms)** | Solo el paso `_model(frame)` sin E/S de cámara ni ROS | Envuelve `process_frame()` con `time.perf_counter()` |
| **Uso de memoria GPU (MB)** | Consumo de VRAM durante inferencia TensorRT | `tegrastats` o `jtop` en Jetson |

### 5.2 Métricas de robustez perceptual

| Métrica | Descripción |
|---------|-------------|
| **Tasa de falsos negativos por clase** | Cuántas señales reales no detecta el modelo por cada tipo (izquierda, stop, etc.) — crítico para seguridad |
| **Blur rejection rate** | Porcentaje de frames rechazados por el filtro de varianza Laplaciana (`_is_blurry`) vs. detecciones perdidas reales |
| **Confidence threshold sensitivity** | Curva de Precision/Recall al variar el umbral de confianza; determina el valor óptimo para el entorno del robot |
| **Estabilidad temporal de detección** | Varianza del `sign_type` detectado en N frames consecutivos para la misma señal (baja varianza = detector estable) |

### 5.3 Métricas de integración con el sistema robótico

| Métrica | Descripción |
|---------|-------------|
| **Tasa de decisiones correctas** | Porcentaje de veces que el robot tomó la acción correcta (giro/stop/avance) en respuesta a una señal real |
| **Distancia de detección efectiva (cm)** | Rango mínimo y máximo al que la señal es detectada con confianza ≥ umbral; relevante para la velocidad de reacción del robot |
| **Tiempo de reacción robot (ms)** | Desde que la señal aparece en frame hasta que el nodo publica la acción correspondiente |

---

## 6. Implementación de YOLO en el sistema de reconocimiento del PuzzleBot

### 6.1 Flujo completo del pipeline

```
Cámara CSI (Jetson)
    │  GStreamer (nvarguscamerasrc → nvvidconv → BGR)
    ▼
camera_node  ──publica──►  /camera/image_raw  (sensor_msgs/Image)
                                    │
                    ┌───────────────┴──────────────────┐
                    ▼                                  ▼
           processor_node                    traffic_light_node
      (YOLO → señales de tráfico)         (HSV → semáforo)
                    │                                  │
                    └──────────────┬───────────────────┘
                                   ▼
                          Acción del robot
```

### 6.2 Carga del modelo (`yolo.py`)

Se prioriza el motor **TensorRT** (`best.engine`) sobre el modelo PyTorch (`best.pt`)
para maximizar el rendimiento en la GPU embebida del Jetson Nano:

```python
if os.path.exists(_ENGINE_PATH):
    MODEL_PATH = _ENGINE_PATH   # ~3-5× más rápido que .pt en Jetson
else:
    MODEL_PATH = _PT_PATH       # Fallback: PyTorch estándar
_model = YOLO(MODEL_PATH, task="detect")
```

### 6.3 Etapas del pipeline de detección

1. **Filtro de calidad (blur check)** — `actividad_2_06.py`
   Calcula la varianza del Laplaciano sobre el frame en escala de grises.
   Si `var < blur_threshold (100.0)`, el frame se descarta sin llamar a YOLO,
   evitando detecciones erróneas en imágenes desenfocadas.

   ```
   score = cv2.Laplacian(gray, cv2.CV_64F).var()
   ```

2. **Inferencia YOLO** — `process_frame()`
   El frame BGR 640×640 se pasa al modelo; se obtienen tensores:
   - `r.boxes.xyxy` → coordenadas `[x1, y1, x2, y2]`
   - `r.boxes.cls`  → ID de clase (0–7)
   - `r.boxes.conf` → confianza (0.0–1.0)

3. **Mapeo de clases a señales** — `get_signs()`
   Las 8 clases del modelo se filtran y remapean a los 6 tipos de señal
   relevantes para la navegación del robot:

   | ID clase | Nombre modelo | ID señal | Acción robot |
   |----------|--------------|----------|--------------|
   | 0 | Forward | 3 | Seguir recto |
   | 2 | Right | 2 | Girar derecha |
   | 7 | left | 1 | Girar izquierda |
   | 5 | Stop | 4 | Detenerse |
   | 1 | GiveWay | 5 | Ceder el paso |
   | 6 | construction | 6 | Zona de obras |

4. **Dibujo de detecciones** — `draw_detections()`
   Sobre el `drawing_frame` (copia del frame original) se dibujan los
   bounding boxes con `cv2.rectangle` y las etiquetas con confianza.
   No modifica el frame original para mantener limpia la inferencia.

5. **Selección de señal frontal** — `_get_front_sign()`
   Entre todas las detecciones válidas, se selecciona la de **mayor área de
   bounding box** como la señal más cercana/prominente al robot.

### 6.4 Detección de semáforo (sin red neuronal)

El nodo `traffic_light_node` usa **segmentación por color HSV** como alternativa
ligera a YOLO para el semáforo, dado que el color es suficiente discriminador:

| Color | Rango HSV bajo | Rango HSV alto | Acción |
|-------|---------------|---------------|--------|
| Rojo | [0,120,70] / [160,120,70] | [10,255,255] / [180,255,255] | Detener robot (`speed=0.0`) |
| Amarillo | [20,100,100] | [35,255,255] | Mantener parado si venía de rojo |
| Verde | [40,50,50] | [80,255,255] | Reanudar marcha (`speed=1.0`) |

Publica en `/speed_multiplier` (Float32) para escalar la velocidad del robot.
