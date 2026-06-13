# miniretoS8 - PuzzleBot Seguidor de Línea con Detección de Zebra

## 📚 Documentación

### 🚦 **[LEER PRIMERO: Guía de Señales y Cruce de Zebra](README_SENALES_ZEBRA.md)**
Documentación completa sobre:
- **3 Casos de cruce de zebra** (sin señal, con señal, con semáforo)
- **Comportamiento de cada señal** (LEFT, RIGHT, FORWARD, CONSTRUCTION, GIVE_WAY, STOP)
- Parámetros configurables
- Variables de estado

---

## 📋 Descripción del proyecto

Sistema robótico basado en ROS2 para seguimiento de línea negra con capacidad de:
- Detección de cruces de zebra
- Reconocimiento de señales de tráfico (YOLO v8)
- Control de semáforo
- Navegación autónoma

---

## 🚀 Ejecución

### Setup (Terminal 1):
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch miniretoS8 setup_launch.py
```

### Autonomy (Terminal 2):
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch miniretoS8 autonomy_launch.py
```

---

## 📁 Estructura

```
miniretoS8/
├── README_SENALES_ZEBRA.md          # 📌 DOCUMENTACIÓN PRINCIPAL
├── miniretoS8/
│   ├── line_follower_node.py         # Nodo principal de control
│   ├── line_detector2.py             # Detección de línea y zebra
│   ├── processor_node.py             # Procesamiento YOLO
│   ├── traffic_light_node.py         # Detección semáforo
│   ├── camera_node.py                # Captura de cámara
│   ├── yolo.py                       # Wrapper YOLO
│   └── simple_camera.py              # Cámara simple
├── launch/
│   ├── setup_launch.py               # Setup: micro_ros_agent, teleop, cámara
│   ├── autonomy_launch.py            # Autonomy: traffic_light, processor, line_follower
│   └── miniretoS8_launch.py          # Lanzador unificado
└── arquitectura_modelo.md             # Arquitectura YOLO
```

---

## 🔧 Compilación

```bash
cd ~/ros2_ws
colcon build --packages-select miniretoS8
source ~/ros2_ws/install/setup.bash
```

---

## 🎯 Comportamiento resumido

| Situación | Qué hace |
|-----------|----------|
| **Línea clara** | Sigue la línea negra con P-controller |
| **Cruce zebra (CASO 1)** | Para 1.5s → Avanza 40cm → Continúa |
| **Cruce + LEFT/RIGHT (CASO 2)** | Para → Avanza 15cm → Gira 90° → Avanza 15cm |
| **Cruce + FORWARD (CASO 2)** | Para → Avanza 40cm → Continúa |
| **Cruce + CONSTRUCTION (CASO 2)** | Sigue línea a 50% por 10cm → Velocidad normal |
| **Cruce + GIVE_WAY (CASO 2)** | Para → Analiza siguiente señal → Ejecuta |
| **Cruce + STOP (CASO 2)** | Para mientras se detecte la señal |
| **Semáforo ROJO (CASO 3)** | Para completamente |
| **Semáforo AMARILLO (CASO 3)** | Si hay señal: para y espera VERDE |
| **Semáforo VERDE (CASO 3)** | Ejecuta acciones guardadas |

---

## 🔍 Referencia rápida

**¿Cómo funciona cada señal?** → Lee [README_SENALES_ZEBRA.md](README_SENALES_ZEBRA.md)

**¿Cuáles son los 3 casos?** → Lee [README_SENALES_ZEBRA.md](README_SENALES_ZEBRA.md) - CASO 1, CASO 2, CASO 3

**¿Qué parámetros puedo cambiar?** → Lee tabla "Parámetros configurables" en [README_SENALES_ZEBRA.md](README_SENALES_ZEBRA.md)

---

## ⚙️ Parámetros principales

- `kp`: 1.5 (ganancia proporcional)
- `linear`: 0.10 m/s (velocidad lineal)
- `max_w`: 2.0 rad/s (velocidad angular máxima)
- `zebra_initial_straight_sec`: 5.56s (avance 40cm)
- `stop_duration`: 1.5s (parada en CASO 1)

---

## 📊 Archivos de configuración

- `line_detector2.py` - Línea 22: `_odd()` para cálculos de kernel
- `line_detector2.py` - Línea 299-303: Parámetros de detección zebra
- `line_follower_node.py` - Línea 39-111: Declaración de parámetros

---

## 🐛 Troubleshooting

### Falsas detecciones de zebra
→ Aumentar `zebra_min_blob_area` o `zebra_roi_y_start`

### Robot oscila mucho (zigzag)
→ Reducir `kp` o aumentar `direction_alpha`

### No detecta construcción
→ Revisar parámetros YOLO en `processor_node.py`

---

## 📝 Última actualización

**Build:** Compilado exitosamente ✅
**Estado:** Listo para pruebas en Jetson

