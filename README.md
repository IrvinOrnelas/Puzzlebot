# PuzzleBot — Navegación Autónoma con Visión

Repositorio del proyecto de robótica móvil sobre la plataforma **PuzzleBot** (Jetson Nano + ROS 2).
El objetivo es lograr navegación autónoma en una pista cerrada con línea guía, cruces peatonales (zebra), señales de tráfico y semáforo.

El paquete **`miniretoS8`** es la entrega final que integra todos los subsistemas.

---

## Estructura del repositorio

```
Puzzlebot/
├── miniretoS8/          ← Entrega final (ver abajo)
├── miniretoS6/          ← Seguimiento de línea básico
├── miniretoS7/          ← Detección de señales con YOLO
├── minireto3/           ← Navegación por waypoints
├── minireto4/           ← Viewer de cámara
├── grabacionpista/      ← Grabación de video de la pista
├── capturafotos/        ← Calibración de cámara (webcam)
└── reto8/               ← Prototipo de detección de pista
```

---

## `miniretoS8` — Entrega Final

Paquete ROS 2 que implementa el pipeline completo de navegación autónoma:
**cámara CSI → detección de línea y zebra → YOLO (señales) → semáforo → control de velocidad**.

### Nodos

| Nodo | Archivo | Función |
|------|---------|---------|
| `camera_node` | [camera_node.py](miniretoS8/miniretoS8/camera_node.py) | Captura frames de la cámara CSI (GStreamer) y los publica en `/camera/image_raw` |
| `processor_node` | [processor_node.py](miniretoS8/miniretoS8/processor_node.py) | Corre inferencia YOLOv8 sobre los frames y publica el tipo de señal detectada |
| `traffic_light_node` | [traffic_light_node.py](miniretoS8/miniretoS8/traffic_light_node.py) | Detecta el estado del semáforo por segmentación HSV (activo solo en cruce zebra) |
| `line_follower_node` | [line_follower_node.py](miniretoS8/miniretoS8/line_follower_node.py) | Controlador principal: integra semáforo, señales y seguimiento de línea con prioridades |

### Topics ROS 2

| Topic | Tipo | Publicador | Descripción |
|-------|------|-----------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | `camera_node` | Frame BGR crudo de la cámara |
| `/speed_multiplier` | `std_msgs/Float32` | `traffic_light_node` | Multiplicador de velocidad: 0.0 (rojo), 0.5 (amarillo), 1.0 (verde) |
| `/stop_sign` | `std_msgs/Float32` | `processor_node` | Área relativa de la señal STOP detectada |
| `/slow_sign` | `std_msgs/Bool` | `processor_node` | Señal de obras/roadwork activa |
| `/giveway_sign` | `std_msgs/Bool` | `processor_node` | Señal de ceda el paso activa |
| `/roundabout_sign` | `std_msgs/Bool` | `processor_node` | Señal de rotonda activa |
| `/sign_command` | `std_msgs/Int32` | `processor_node` | Comando direccional: 0=nada, 1=izq, 2=der, 3=adelante |
| `/at_zebra` | `std_msgs/Bool` | `line_follower_node` | Indica al semáforo si el robot está en un cruce zebra |
| `/cmd_vel` | `geometry_msgs/Twist` | `line_follower_node` | Velocidad lineal y angular enviada al robot |

### Arquitectura del sistema

```
Cámara CSI (GStreamer nvarguscamerasrc)
    │
    ▼
camera_node ──► /camera/image_raw
                    │
        ┌───────────┼───────────────┐
        ▼           ▼               ▼
processor_node  traffic_light_node  line_follower_node
  (YOLO)          (HSV semáforo)    (línea + zebra)
        │               │               │
        │   /stop_sign  │ /speed_mult   │
        │   /sign_cmd   │               │ /at_zebra
        └───────────────┴───────────────┘
                        ▼
                   /cmd_vel → Robot
```

### Lógica de prioridades en `line_follower_node`

El controlador aplica las siguientes prioridades en cada frame:

1. **Semáforo** (`/speed_multiplier`) — Si `mult ≤ 0.01`, parada total inmediata.
2. **Señal STOP** (`/stop_sign`) — Si el área supera el umbral, parada definitiva.
3. **Cruce zebra** — FSM de 3 fases (STOP → ADVANCE → ACTION) con soporte para:
   - *Caso 1:* Sin señal pendiente → parar 1.5 s y avanzar recto.
   - *Caso 2:* Con señal direccional → parar, avanzar y ejecutar el giro.
   - *Caso 3:* Con señal y semáforo → esperar verde, luego avanzar y girar.
4. **Seguimiento de línea** — Controlador P con filtros de suavizado y slew-rate.

### Modelo YOLO

- **Arquitectura:** YOLOv8 (CSPDarknet + PAN-FPN + head desacoplado)
- **Weights:** [`best.pt`](miniretoS8/miniretoS8/best.pt) — también acepta `best.engine` (TensorRT) para mayor rendimiento en Jetson Nano
- **Dataset:** SENAS2, 8 clases → mapeadas a 6 señales de interés
- **Métricas de validación (mejor epoch):**

| Métrica | Valor |
|---------|-------|
| Precision | 98.4 % |
| Recall | 95.1 % |
| mAP@50 | 98.75 % |
| mAP@50-95 | 75.77 % |

Mapeo de clases:

| ID clase | Nombre | Acción robot |
|----------|--------|--------------|
| 0 | Forward | Seguir recto |
| 2 | Right | Girar derecha |
| 7 | left | Girar izquierda |
| 5 | Stop | Detenerse |
| 1 | GiveWay | Ceder el paso |
| 6 | construction | Reducir velocidad |
| 3 | Roundabout | Velocidad rotonda |

---

## Dependencias

- ROS 2 (Foxy / probado en Jetson Nano con Ubuntu 18.04 + ROS 2 Foxy)
- Python 3.8
- `rclpy`, `sensor_msgs`, `cv_bridge`, `geometry_msgs`, `std_msgs`
- `opencv-python`
- `ultralytics` (YOLOv8)
- GStreamer con soporte NVMM (Jetson)

---

## Instalación y build

```bash
# Desde la raíz del workspace
cd ~/ros2_ws
colcon build --packages-select miniretoS8
source install/setup.bash
```

---

## Ejecución

### Nodos individuales

```bash
ros2 run miniretoS8 camera_node
ros2 run miniretoS8 processor_node
ros2 run miniretoS8 traffic_light_node
ros2 run miniretoS8 line_follower_node
```

---

## Calibración de cámara

Los parámetros intrínsecos usados en la detección de zebra y línea se encuentran embebidos en [track_detector.py](miniretoS8/miniretoS8/track_detector.py).
Para recalibrar con otra cámara, utilizar los scripts en [capturafotos/](capturafotos/):

```bash
# Capturar imágenes del tablero de ajedrez
python3 capturafotos/capturar_webcam.py

# Calcular parámetros de calibración
python3 capturafotos/calibrar_webcam.py
```

---

## Otros paquetes del repositorio

| Paquete | Descripción |
|---------|-------------|
| `miniretoS6` | Seguidor de línea sin señales ni semáforo |
| `miniretoS7` | Detección de señales con YOLO (sin integración de zebra/semáforo) |
| `minireto3` | Navegación por waypoints con estimación de pose por odometría |
| `minireto4` | Nodo viewer de la cámara |
| `grabacionpista` | Grabación de video de sesiones de prueba |
| `reto8` | Prototipo de detección de pista con umbral adaptativo |
