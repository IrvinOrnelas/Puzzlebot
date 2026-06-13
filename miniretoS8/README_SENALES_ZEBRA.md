# Sistema de Detección de Señales y Cruce de Zebra - PuzzleBot miniretoS8

## 📋 Descripción General

Este documento especifica el comportamiento del robot en los **3 casos de cruce de zebra** y el comportamiento de cada **señal de tráfico** implementada.

**Prioridad general:** Semáforo > Señales > Seguidor de Línea

---

## 🚦 CASO 1: Solo Cruce Zebra (Sin señal ni semáforo)

### Flujo de ejecución:
```
Detecta zebra → Parada 1.5s → Avanza 40cm recto → Búsqueda de línea
```

### Detalles:
1. Robot detecta cruce de zebra en la ROI (inferior 25% de imagen)
2. **Se detiene completamente (v=0) durante 1.5 segundos** (parada de seguridad)
3. **Avanza 40cm en línea recta** sin buscar línea (solo movimiento recto)
4. Vuelve a modo búsqueda de línea normal
5. Contador interno: `0 → 1` (espera segundo cruce)
6. Cuando detecta segundo cruce: `1 → 0` (reinicia ciclo)

### Variables involucradas:
- `_zebra_crossing_counter`: 0 (primer cruce) / 1 (segundo cruce)
- `_zebra_active`: True durante la ejecución
- `_zebra_phase_start`: Timestamp de inicio

### Tiempo total: ~8 segundos
- Parada: 1.5s
- Avance 40cm: 5.56s (a 0.072 m/s)
- Margen: 1s

---

## 🚩 CASO 2: Cruce Zebra + Señal (Sin semáforo)

### Lógica principal:
**La señal se GUARDA cuando se detecta**, pero se ejecuta **SOLO cuando se detecta el cruce de zebra**.

### Señal 1: LEFT (Giro Izquierda)

**¿Qué hace?**
```
Avanza 15cm → Gira 90° izquierda → Avanza 15cm → Búsqueda línea
```

**Detalles:**
- **Fase 1:** Avanza 15cm en línea recta (sin girar)
- **Fase 2:** Gira exactamente 90° sobre su propio eje (giro izquierda)
- **Fase 3:** Avanza 15cm en línea recta (dirección nueva)
- Vuelve a búsqueda de línea normal

**Tiempos:**
- Avance 15cm: ~2.08s (15cm ÷ 0.072 m/s)
- Giro 90°: ~5.23s (1.57 rad ÷ 0.30 rad/s)
- Avance 15cm: ~2.08s
- **Total: ~9.4 segundos**

**Código relevante:**
```python
action == 1  # LEFT
_zebra_turn_subphase: 'ADVANCE_1' → 'TURN' → 'ADVANCE_2'
omega = 0.30  # rad/s para giro izquierda
```

---

### Señal 2: RIGHT (Giro Derecha)

**¿Qué hace?**
```
Avanza 15cm → Gira 90° derecha → Avanza 15cm → Búsqueda línea
```

**Idéntico a LEFT pero gira a la derecha**

**Detalles:**
- **Fase 1:** Avanza 15cm en línea recta
- **Fase 2:** Gira exactamente 90° sobre su propio eje (giro derecha)
- **Fase 3:** Avanza 15cm en línea recta (dirección nueva)

**Tiempos:** Idénticos a LEFT (~9.4 segundos)

**Código relevante:**
```python
action == 2  # RIGHT
omega = -0.30  # rad/s para giro derecha (negativo = derecha)
```

---

### Señal 3: FORWARD (Adelante)

**¿Qué hace?**
```
Avanza 40cm recto → Búsqueda línea
```

**Detalles:**
- Avanza **40cm en línea recta** (sin seguir línea, solo avanzar)
- No gira
- Vuelve a búsqueda de línea normal

**Tiempo:**
- Avance 40cm: ~5.56s (40cm ÷ 0.072 m/s)
- **Total: ~5.56 segundos**

**Código relevante:**
```python
action == 3  # FORWARD
v = self._linear * self._traffic_mult * scales
```

---

### Señal 4: CONSTRUCTION (Construcción/Zona de Obras)

**¿Qué hace?**
```
Sigue línea a 50% velocidad por 10cm (~3s) → Vuelve a velocidad normal
```

**Detalles:**
- **NO es una acción de movimiento recto**
- **Sigue la línea** igual que modo normal, pero a **50% de velocidad**
- Dura aproximadamente **3 segundos** (tiempo para recorrer ~10cm a 50%)
- **Nunca deja de seguir la línea**, simplemente la sigue más lentamente
- Después de 3s, desactiva flag y vuelve a velocidad normal

**Velocidad:**
- Normal: 0.072 m/s (0.1 × 0.72)
- CONSTRUCTION: 0.036 m/s (0.1 × 0.72 × 0.5)
- Distancia: 10cm
- Tiempo: 10cm ÷ 3.6 cm/s ≈ 2.78s (~3s)

**Variables:**
- `_construction_active`: True mientras esté en zona
- `_construction_start_time`: Timestamp de inicio
- Duration: 3.0 segundos

**Código relevante:**
```python
action == 4  # CONSTRUCTION
self._construction_active = True
# En seguidor de línea:
if self._construction_active:
    speed_scale = 0.5
```

---

### Señal 5: GIVE_WAY (Ceda/Yield)

**¿Qué hace?**
```
Se detiene (v=0) → Analiza siguientes señales (con o sin semáforo) → Ejecuta si hay
```

**Detalles:**
- Cuando se detecta GIVE_WAY y se llega al cruce de zebra
- Robot se **detiene completamente (v=0)**
- **Analiza si hay más señales** próximas
  - Puede analizar señales sin semáforo
  - Puede analizar señales con semáforo (respeta colores)
- Si detecta nueva señal, la ejecuta
- Si no detecta, continúa en búsqueda normal

**Caso de uso:** Zona de cedencia, donde hay que ceder y luego esperar siguiente acción

**Código relevante:**
```python
action == 5  # GIVE_WAY
status = 'ZEBRA-DONE-GIVEWAY'
# Robot se detiene y espera análisis de siguiente señal
```

---

### Señal 6: STOP (Alto)

**¿Qué hace?**
```
Se detiene (v=0) mientras se detecte → Vuelve a normal cuando desaparece
```

**Detalles:**
- Cuando se detecta STOP en cruce de zebra
- Robot se **detiene completamente (v=0)**
- **Mantiene parada mientras se detecte la señal**
- Cuando **deja de detectar** la señal, vuelve a modo normal
- **NO es parada definitiva** (a diferencia del semáforo rojo)

**Diferencia con semáforo rojo:**
- STOP sign: parada mientras se detecte (temporal)
- Semáforo rojo: parada mientras esté rojo (puede ser largo)

**Código relevante:**
```python
action == 6  # STOP
# Se detiene mientras se mantenga detectado
# Cuando deja de detectarse, vuelve a normal
```

---

## 🚦 CASO 3: Cruce Zebra + Señal + Semáforo

### Lógica de semáforo (Prioridad máxima)

```
ROJO: v=0 (parada total)
AMARILLO: Si hay señal → detiene y espera VERDE; Si no → continúa
VERDE: Ejecuta acciones (igual que CASO 2)
```

### SEMÁFORO ROJO

**¿Qué hace?**
- Robot **detenido (v=0)** completamente
- **Ignora todo** (señales, línea, zebra)
- Espera a que cambie a amarillo o verde

**Variables:**
```python
traffic_color == 'RED'
v = 0.0  # Parada total
```

---

### SEMÁFORO AMARILLO

**¿Qué hace?**
```
Si detecta señal:
  → Se detiene (v=0)
  → Espera siguiente VERDE
  → Ejecuta la acción guardada

Si NO detecta señal:
  → Continúa búsqueda línea
```

**Detalles:**
- Si robot detecta una **SEÑAL mientras está en AMARILLO**:
  - Se **detiene (v=0)**
  - **Guarda la señal**
  - **Espera a que salga VERDE**
  - Cuando es VERDE: ejecuta la acción
- Si **NO hay señal**:
  - Continúa en modo búsqueda de línea
  - Sin cambios especiales

**Código relevante:**
```python
traffic_color == 'YELLOW'
if action in (1, 2, 3, 4, 5, 6):  # Hay señal
    v = 0.0
    status = 'ZEBRA-SEMAF-AMARILLO-ESPERA'
else:  # No hay señal
    v = 0.0
    status = 'ZEBRA-SEMAF-AMARILLO'
```

---

### SEMÁFORO VERDE

**¿Qué hace?**
- **Ejecuta las acciones guardadas** de la señal
- Comportamiento idéntico a CASO 2
- LEFT, RIGHT, FORWARD, CONSTRUCTION, GIVE_WAY, STOP funcionan normalmente

**Código relevante:**
```python
traffic_color == 'GREEN'
v, omega, status = self._execute_zebra_action(action, phase_elapsed, now)
```

---

## 📊 Tabla resumen de señales

| Señal | Acción | Tiempo total | Avanza | Gira | Velocidad |
|-------|--------|--------------|--------|------|-----------|
| FORWARD | Avanza 40cm | 5.56s | 40cm | - | 100% |
| LEFT | Avanza 15 + gira 90° + avanza 15 | 9.4s | 30cm | SÍ (izq) | 100% |
| RIGHT | Avanza 15 + gira 90° + avanza 15 | 9.4s | 30cm | SÍ (der) | 100% |
| CONSTRUCTION | Sigue línea 10cm | 3.0s | 10cm | Sigue línea | 50% |
| GIVE_WAY | Se detiene | Variable | - | - | 0% |
| STOP | Se detiene | Variable | - | - | 0% |

---

## 🔧 Parámetros configurables

Ubicación: `/home/puzzlebot/ros2_ws/src/miniretoS8/miniretoS8/line_follower_node.py`

| Parámetro | Valor actual | Descripción |
|-----------|--------------|-------------|
| `zebra_initial_straight_sec` | 5.56 | Tiempo para avanzar 40cm (CASO 1 y FORWARD) |
| `zebra_roi_y_start` | 0.75 | ROI de detección zebra: 0.75 = inferior 25% |
| `zebra_min_blob_area` | 200.0 | Área mínima de blob para aceptar detección |
| `stop_duration` (CASO 1) | 1.5 | Duración parada inicial en cruce |
| `_construction_duration` | 3.0 | Duración en zona de construcción (10cm) |
| `_turn_omega` | 0.30 | Velocidad angular giros LEFT/RIGHT (rad/s) |
| `_linear` | 0.10 | Velocidad lineal base (m/s) |
| `_zebra_speed` | 0.72 | Multiplicador de velocidad en acciones zebra |

---

## 🎯 Variables de estado principales

```python
_zebra_active              # True si ejecutando acción de zebra
_zebra_crossing_counter    # 0 (primer cruce) o 1 (segundo cruce)
_zebra_action_type         # 1-6 (tipo de señal: LEFT, RIGHT, FORWARD, etc)
_zebra_has_signal          # True si hay señal pendiente
_zebra_has_traffic         # True si hay semáforo
_zebra_traffic_color       # 'RED', 'YELLOW', 'GREEN'
_zebra_turn_subphase       # 'NONE', 'ADVANCE_1', 'TURN', 'ADVANCE_2' (para LEFT/RIGHT)
_construction_active       # True si en zona de construcción
_construction_start_time   # Timestamp inicio construcción
```

---

## 📝 Últimos cambios implementados

✅ **Parada CASO 1:** Aumentada de 0.5s a **1.5s**
✅ **LEFT/RIGHT:** Implementado con 3 fases exactas (Avanza 15 → Gira 90° → Avanza 15)
✅ **CONSTRUCTION:** Sigue línea a **50%** por **10cm** (~3s), nunca pierde línea
✅ **CASO 3 AMARILLO:** Si detecta señal → detiene y espera **VERDE**
✅ **Erosión zebra_mask:** Aumentada a **7x5 con 2 iteraciones** para evitar falsos positivos
✅ **Ciclo 2-cruces:** Primer cruce ejecuta, segundo reinicia contador

---

## 🔍 Archivos relacionados

- `line_follower_node.py` - Nodo principal con lógica de control
- `line_detector2.py` - Detección de línea y cruce de zebra
- `processor_node.py` - Procesamiento de señales YOLO
- `traffic_light_node.py` - Detección de semáforo
- `arquitectura_modelo.md` - Arquitectura del modelo YOLO

---

## 💡 Notas importantes

1. **Ciclo de 2 cruces:** El sistema detecta 2 cruces consecutivos (diseño de pista). El primero se ejecuta, el segundo reinicia el contador.

2. **CONSTRUCTION no es acción:** A diferencia de LEFT/RIGHT/FORWARD, CONSTRUCTION no genera movimiento automático. Solo reduce velocidad mientras el seguidor de línea controla.

3. **Prioridad:** En CASO 3, el semáforo tiene máxima prioridad. Si está en rojo, todo se detiene.

4. **Análisis de señales:** En CASO 2, GIVE_WAY permite analizar siguientes señales (con o sin semáforo), lo que permite encadenar acciones.

5. **Stop no es definitivo:** A diferencia de semáforo rojo, STOP se detiene solo mientras se detecte, permitiendo continuar cuando desaparece.

