import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32

from miniretoS8.line_detector2 import find_line, find_zebra


WINDOW_TITLE = 'Line Follower Priority'
_PROC_SCALE = 0.5

SIGN_NAMES = {0: 'none', 1: 'left', 2: 'right', 3: 'forward'}


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class LineFollowerNode(Node):
    """
    Control final con prioridades:

    1) Semáforo (/speed_multiplier): rojo detiene todo.
    2) Señales YOLO (/stop_sign, /slow_sign, /giveway_sign, /roundabout_sign, /sign_command).
    3) Seguidor de línea y guardia de paso zebra.

    Regla importante: si hay zebra pero NO hay señal direccional pendiente, sigue recto.
    Así la cebra no vuelve a contaminar la dirección del seguidor.
    """

    def __init__(self):
        super().__init__('line_follower_node')

        # Parámetros equivalentes a line_follower.py (controlador P simple).
        self.declare_parameter('kp', 1.5)
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('linear', 0.10)
        self.declare_parameter('max_w', 2.0)
        self.declare_parameter('direction_alpha', 0.35)
        self.declare_parameter('direction_slew_rate', 10.0)
        self.declare_parameter('omega_alpha', 1.0)
        self.declare_parameter('omega_slew_rate', 10.0)
        self.declare_parameter('deriv_limit', 0.0)
        self.declare_parameter('error_deadband', 0.0)
        self.declare_parameter('lost_speed_scale', 0.65)
        self.declare_parameter('stop_when_lost', False)
        self.declare_parameter('show_window', True)

        # Zebra: si no hay señal, ir recto.
        self.declare_parameter('zebra_speed', 0.72)
        self.declare_parameter('zebra_initial_straight_sec', 5.56)
        self.declare_parameter('zebra_extra_advance_sec', 1.39)
        self.declare_parameter('zebra_validation_timeout', 2.0)
        self.declare_parameter('zebra_advance_after_signal_sec', 2.08)
        self.declare_parameter('zebra_forward_action_sec', 2.78)
        self.declare_parameter('zebra_roi_y_start', 0.75)
        self.declare_parameter('zebra_min_blob_area', 200.0)
        self.declare_parameter('zebra_debounce_frames', 1)
        self.declare_parameter('zebra_clear_sec', 0.75)

        # Señales YOLO.
        self.declare_parameter('sign_hold_sec', 1.25)
        self.declare_parameter('pending_turn_timeout', 6.0)
        self.declare_parameter('enable_turn_signs', True)
        self.declare_parameter('execute_turn_only_on_zebra', True)
        self.declare_parameter('turn_hold', 1.10)
        self.declare_parameter('turn_omega', 0.30)
        self.declare_parameter('turn_linear', 0.025)
        self.declare_parameter('slow_speed_scale', 0.55)
        self.declare_parameter('giveway_speed_scale', 0.50)
        self.declare_parameter('roundabout_speed_scale', 0.40)

        self._kp = float(self.get_parameter('kp').value)
        self._kd = float(self.get_parameter('kd').value)
        self._linear = float(self.get_parameter('linear').value)
        self._max_w = float(self.get_parameter('max_w').value)
        self._alpha = float(self.get_parameter('direction_alpha').value)
        self._direction_slew_rate = float(self.get_parameter('direction_slew_rate').value)
        self._omega_alpha = float(self.get_parameter('omega_alpha').value)
        self._omega_slew_rate = float(self.get_parameter('omega_slew_rate').value)
        self._deriv_limit = float(self.get_parameter('deriv_limit').value)
        self._deadband = float(self.get_parameter('error_deadband').value)
        self._lost_scale = float(self.get_parameter('lost_speed_scale').value)
        self._stop_when_lost = bool(self.get_parameter('stop_when_lost').value)
        self._show_window = bool(self.get_parameter('show_window').value)

        self._zebra_speed = float(self.get_parameter('zebra_speed').value)
        self._zebra_initial_straight_sec = float(self.get_parameter('zebra_initial_straight_sec').value)
        self._zebra_extra_advance_sec = float(self.get_parameter('zebra_extra_advance_sec').value)
        self._zebra_validation_timeout = float(self.get_parameter('zebra_validation_timeout').value)
        self._zebra_advance_after_signal_sec = float(self.get_parameter('zebra_advance_after_signal_sec').value)
        self._zebra_forward_action_sec = float(self.get_parameter('zebra_forward_action_sec').value)
        self._zebra_roi_y_start = float(self.get_parameter('zebra_roi_y_start').value)
        self._zebra_min_blob_area = float(self.get_parameter('zebra_min_blob_area').value)
        self._zebra_debounce_frames = int(self.get_parameter('zebra_debounce_frames').value)
        self._zebra_clear_sec = float(self.get_parameter('zebra_clear_sec').value)

        self._sign_hold_sec = float(self.get_parameter('sign_hold_sec').value)
        self._pending_turn_timeout = float(self.get_parameter('pending_turn_timeout').value)
        self._enable_turn_signs = bool(self.get_parameter('enable_turn_signs').value)
        self._execute_turn_only_on_zebra = bool(self.get_parameter('execute_turn_only_on_zebra').value)
        self._turn_hold = float(self.get_parameter('turn_hold').value)
        self._turn_omega = float(self.get_parameter('turn_omega').value)
        self._turn_linear = float(self.get_parameter('turn_linear').value)
        self._slow_speed_scale = float(self.get_parameter('slow_speed_scale').value)
        self._giveway_speed_scale = float(self.get_parameter('giveway_speed_scale').value)
        self._roundabout_speed_scale = float(self.get_parameter('roundabout_speed_scale').value)

        self._bridge = CvBridge()

        # Filtros del seguidor de línea.
        self._filtered_dir = 0.0
        self._last_valid_dir = 0.0
        self._prev_filtered_dir = 0.0
        self._raw_dir_filtered = 0.0
        self._omega_filtered = 0.0
        self._last_control_time = self.get_clock().now()
        self._last_line_time = self.get_clock().now()

        # Prioridad 1: semáforo.
        self._traffic_mult = 1.0

        # Prioridad 2: señales.
        self._stop_sign_active = False
        self._stop_area_ratio = 0.0
        self._final_stop = False
        self._STOP_SLOW_THRESH = 0.03
        self._STOP_FULL_THRESH = 0.08

        self._slow_sign_active = False
        self._giveway_sign_active = False
        self._roundabout_active = False
        self._giveway_pending = False

        self._sign_command = 0
        self._sign_cmd_last_time = None
        self._pending_turn = 0
        self._pending_turn_time = None
        self._turn_active_cmd = 0
        self._turn_start_time = None

        # Estado zebra.
        self._zebra_seen_frames = 0
        self._zebra_hold_start = None
        self._zebra_last_seen = None
        self._zebra_active = False
        self._zebra_phase = 'NONE'
        self._zebra_phase_start = None
        self._zebra_action_type = None
        self._zebra_has_signal = False
        self._zebra_has_traffic = False
        self._zebra_traffic_color = None
        self._zebra_crossing_counter = 0

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub = self.create_subscription(Image, '/camera/image_raw', self._image_callback, qos)
        self._traffic_sub = self.create_subscription(Float32, '/speed_multiplier', self._traffic_callback, 10)
        self._stop_sub = self.create_subscription(Float32, '/stop_sign', self._stop_sign_callback, 10)
        self._slow_sub = self.create_subscription(Bool, '/slow_sign', self._slow_sign_callback, 10)
        self._giveway_sub = self.create_subscription(Bool, '/giveway_sign', self._giveway_callback, 10)
        self._roundabout_sub = self.create_subscription(Bool, '/roundabout_sign', self._roundabout_callback, 10)
        self._sign_cmd_sub = self.create_subscription(Int32, '/sign_command', self._sign_cmd_callback, 10)

        if self._show_window:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

        self.get_logger().info('LineFollower priority listo: semáforo > señales > línea')

    # ------------------------------------------------------------------
    def _publish_stop(self):
        self._cmd_pub.publish(Twist())

    def _seconds_since(self, stamp):
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def _publish_cmd(self, v, omega):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self._cmd_pub.publish(cmd)

    def _reset_steering_filters(self):
        self._filtered_dir = 0.0
        self._last_valid_dir = 0.0
        self._prev_filtered_dir = 0.0
        self._raw_dir_filtered = 0.0
        self._omega_filtered = 0.0

    # ------------------------------------------------------------------
    # Callbacks de prioridades
    def _traffic_callback(self, msg: Float32):
        self._traffic_mult = float(msg.data)

    def _stop_sign_callback(self, msg: Float32):
        ratio = float(msg.data)
        was_active = self._stop_sign_active
        self._stop_area_ratio = ratio
        self._stop_sign_active = ratio >= self._STOP_FULL_THRESH
        if self._stop_sign_active and not was_active:
            self._final_stop = True
            self.get_logger().info(f'STOP SIGN FINAL: parada definitiva activada (area={ratio:.3f})')
            self._publish_stop()

    def _slow_sign_callback(self, msg: Bool):
        self._slow_sign_active = bool(msg.data)

    def _giveway_callback(self, msg: Bool):
        self._giveway_sign_active = bool(msg.data)
        if msg.data and not self._giveway_pending:
            self._giveway_pending = True
            self.get_logger().info('Señal CEDA/YIELD → reducir hasta zebra')

    def _roundabout_callback(self, msg: Bool):
        self._roundabout_active = bool(msg.data)

    def _sign_cmd_callback(self, msg: Int32):
        cmd = int(msg.data)
        now = self.get_clock().now()

        if cmd == 3:
            self._sign_command = 3
            self._sign_cmd_last_time = now
            return

        if cmd in (1, 2) and self._enable_turn_signs:
            if self._execute_turn_only_on_zebra:
                if cmd != self._pending_turn:
                    self.get_logger().info(f'Señal {SIGN_NAMES[cmd]} detectada → vuelta pendiente hasta zebra')
                self._pending_turn = cmd
                self._pending_turn_time = now
            else:
                self._start_turn(cmd)
            return

        if cmd == 0:
            # No borrar pending_turn inmediatamente; se mantiene unos segundos para llegar a la zebra.
            if self._sign_command == 3:
                self._sign_command = 0
                self._sign_cmd_last_time = None

    # ------------------------------------------------------------------
    def _sign_forward_active(self):
        return (
            self._sign_command == 3
            and self._sign_cmd_last_time is not None
            and self._seconds_since(self._sign_cmd_last_time) <= self._sign_hold_sec
        )

    def _pending_turn_valid(self):
        return (
            self._pending_turn in (1, 2)
            and self._pending_turn_time is not None
            and self._seconds_since(self._pending_turn_time) <= self._pending_turn_timeout
        )

    def _start_turn(self, cmd: int):
        self._turn_active_cmd = int(cmd)
        self._turn_start_time = self.get_clock().now()
        self._pending_turn = 0
        self._pending_turn_time = None
        self._zebra_active = False
        self._zebra_hold_start = None
        self._reset_steering_filters()
        self.get_logger().info(f'Ejecutando vuelta {SIGN_NAMES.get(cmd, cmd)}')

    def _turn_active(self):
        return (
            self._turn_active_cmd in (1, 2)
            and self._turn_start_time is not None
            and self._seconds_since(self._turn_start_time) <= self._turn_hold
        )

    def _finish_turn_if_needed(self):
        if self._turn_active_cmd in (1, 2) and self._turn_start_time is not None:
            if self._seconds_since(self._turn_start_time) > self._turn_hold:
                self.get_logger().info('Vuelta completada → retomar línea')
                self._turn_active_cmd = 0
                self._turn_start_time = None
                self._reset_steering_filters()

    def _sign_speed_scale(self):
        # Señales de velocidad reducida. Orden: rotonda más restrictiva.
        if self._roundabout_active:
            return self._roundabout_speed_scale, 'ROUNDABOUT'
        if self._giveway_pending or self._giveway_sign_active:
            return self._giveway_speed_scale, 'GIVEWAY'
        if self._slow_sign_active:
            return self._slow_speed_scale, 'SLOW'
        return 1.0, ''

    def _show_and_maybe_exit(self, vis, text):
        if not self._show_window:
            return
        cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow(WINDOW_TITLE, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            raise SystemExit

    # ------------------------------------------------------------------
    def _image_callback(self, msg: Image):
        # PRIORIDAD 1: SEMÁFORO. Rojo manda sobre todo.
        if self._traffic_mult <= 0.01:
            self._publish_stop()
            return

        # PRIORIDAD 2: señales de STOP.
        if self._final_stop or self._stop_sign_active:
            self._publish_stop()
            return

        stop_approach = self._STOP_SLOW_THRESH <= self._stop_area_ratio < self._STOP_FULL_THRESH
        stop_approach_scale = 0.4 if stop_approach else 1.0
        sign_speed_scale, sign_tag = self._sign_speed_scale()

        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self._show_window and cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
            raise SystemExit

        small = cv2.resize(frame, None, fx=_PROC_SCALE, fy=_PROC_SCALE, interpolation=cv2.INTER_LINEAR)

        # Detectar zebra antes que línea. Si hay zebra y no hay señal, ir recto.
        zebra_vis, zebra_detected, zebra_y_pos = find_zebra(
            small,
            roi_y_start=self._zebra_roi_y_start,
            min_blob_area=self._zebra_min_blob_area,
        )

        now = self.get_clock().now()
        if zebra_detected:
            self._zebra_seen_frames += 1
            self._zebra_last_seen = now
            if self._giveway_pending:
                self._giveway_pending = False
                self.get_logger().info('Zebra alcanzada tras CEDA/YIELD → cancelar pendiente')
        else:
            self._zebra_seen_frames = 0

        # Si hay una vuelta pendiente y se detectó zebra, ahora sí ejecutar señal.
        if zebra_detected and self._zebra_seen_frames >= self._zebra_debounce_frames and self._pending_turn_valid():
            self._start_turn(self._pending_turn)

        # Expirar vuelta pendiente si nunca apareció zebra.
        if self._pending_turn in (1, 2) and not self._pending_turn_valid():
            self.get_logger().info(f'Vuelta pendiente {SIGN_NAMES.get(self._pending_turn)} expirada → ignorar')
            self._pending_turn = 0
            self._pending_turn_time = None

        # Señal direccional activa: prioridad sobre línea/zebra.
        self._finish_turn_if_needed()
        if self._turn_active():
            turn_sign = 1.0 if self._turn_active_cmd == 1 else -1.0
            omega = turn_sign * self._turn_omega
            v = self._turn_linear * self._traffic_mult * stop_approach_scale * sign_speed_scale
            self._publish_cmd(v, omega)
            status = f'SIGN-{SIGN_NAMES[self._turn_active_cmd].upper()}'
            vis = cv2.resize(zebra_vis, (frame.shape[1], frame.shape[0])) if self._show_window else None
            if vis is not None:
                self._show_and_maybe_exit(vis, f'{status} {sign_tag} v={v:.2f} w={omega:+.2f}')
            self.get_logger().info(f'{status} | v={v:.3f} w={omega:+.3f}')
            return

        # Forward sign: seguir recto por encima del seguidor de línea.
        if self._sign_forward_active():
            self._reset_steering_filters()
            v = self._linear * self._traffic_mult * stop_approach_scale * sign_speed_scale
            self._publish_cmd(v, 0.0)
            status = 'SIGN-FORWARD'
            vis = cv2.resize(zebra_vis, (frame.shape[1], frame.shape[0])) if self._show_window else None
            if vis is not None:
                self._show_and_maybe_exit(vis, f'{status} {sign_tag} v={v:.2f} w=0.00')
            self.get_logger().info(f'{status} | v={v:.3f} w=+0.000')
            return

        # ZEBRA: Gestión con ciclo de 2 cruces (reinicia cada 2 detecciones)
        # Si se detectó el segundo cruce (validación), reiniciar contador para el siguiente ciclo
        if self._zebra_crossing_counter == 1 and zebra_detected and not self._zebra_active:
            self._zebra_crossing_counter = 0
            self._zebra_seen_frames = 0  # Reset frames para NO entrar a inicializar nuevamente
            self.get_logger().info('Segundo cruce detectado → reiniciar contador, próximo ciclo')

        if self._zebra_seen_frames >= self._zebra_debounce_frames and not self._zebra_active and self._zebra_crossing_counter == 0:
            has_signal = self._pending_turn_valid()
            has_traffic = self._traffic_mult < 1.0
            self._zebra_active = True
            self._zebra_has_signal = has_signal
            self._zebra_has_traffic = has_traffic
            self._zebra_action_type = self._pending_turn if has_signal else 0
            self._zebra_phase = 'ADVANCE'
            self._zebra_phase_start = now
            case = 3 if (has_signal and has_traffic) else (2 if has_signal else 1)
            self.get_logger().info(f'ZEBRA detectada y={zebra_y_pos:+.2f} → CASO {case}')

        if self._zebra_active:
            phase_elapsed = self._seconds_since(self._zebra_phase_start) if self._zebra_phase_start else 0.0
            status = 'ZEBRA'
            v = 0.0
            omega = 0.0

            # CASO 1: Avanza 30cm, espera validación (2º cruce), si no hay avanza 10cm más
            if not self._zebra_has_signal and not self._zebra_has_traffic:
                status = 'ZEBRA-CASO1'

                if phase_elapsed < self._zebra_initial_straight_sec:
                    # Fase 1: Avanza 30cm
                    v = self._linear * self._zebra_speed * self._traffic_mult * stop_approach_scale * sign_speed_scale
                    status = 'ZEBRA-AVANCE-1'
                elif phase_elapsed < self._zebra_initial_straight_sec + self._zebra_validation_timeout:
                    # Fase 2: Espera segundo cruce (validación)
                    status = 'ZEBRA-VALIDANDO'
                    v = 0.0
                elif phase_elapsed < self._zebra_initial_straight_sec + self._zebra_validation_timeout + self._zebra_extra_advance_sec:
                    # Fase 3: No detectó segundo cruce, avanza 10cm más
                    v = self._linear * self._zebra_speed * self._traffic_mult * stop_approach_scale * sign_speed_scale
                    status = 'ZEBRA-AVANCE-2'
                else:
                    # Finaliza primer cruce - incrementar contador para esperar segundo
                    self._zebra_active = False
                    self._zebra_phase = 'NONE'
                    self._zebra_crossing_counter = 1
                    self.get_logger().info('ZEBRA-CASO1 completado → esperar segundo cruce')

            # CASO 2 y 3: Avanza + acción (con señal ± semáforo)
            else:
                if self._zebra_phase == 'ADVANCE':
                    if phase_elapsed < self._zebra_advance_after_signal_sec:
                        status = 'ZEBRA-AVANCE'
                        v = self._linear * self._zebra_speed * self._traffic_mult * stop_approach_scale * sign_speed_scale
                    else:
                        self._zebra_phase = 'ACTION'
                        self._zebra_phase_start = now

                elif self._zebra_phase == 'ACTION':
                    action = self._zebra_action_type
                    status = 'ZEBRA-ACTION'

                    if action == 1:  # LEFT
                        giro_time = 1.57 / max(0.1, self._turn_omega)
                        if phase_elapsed < giro_time:
                            status = 'ZEBRA-LEFT'
                            omega = self._turn_omega
                        else:
                            self._zebra_active = False
                            self._zebra_phase = 'NONE'
                    elif action == 2:  # RIGHT
                        giro_time = 1.57 / max(0.1, self._turn_omega)
                        if phase_elapsed < giro_time:
                            status = 'ZEBRA-RIGHT'
                            omega = -self._turn_omega
                        else:
                            self._zebra_active = False
                            self._zebra_phase = 'NONE'
                    elif action == 3:  # FORWARD
                        if phase_elapsed < self._zebra_forward_action_sec:
                            status = 'ZEBRA-FORWARD'
                            v = self._linear * self._traffic_mult * stop_approach_scale * sign_speed_scale
                        else:
                            self._zebra_active = False
                            self._zebra_phase = 'NONE'
                    elif action == 4:  # CONSTRUCTION
                        if phase_elapsed < 1.39:
                            status = 'ZEBRA-CONSTR'
                            v = self._linear * 0.6 * self._traffic_mult * stop_approach_scale * sign_speed_scale
                        else:
                            self._zebra_active = False
                            self._zebra_phase = 'NONE'
                    elif action == 5:  # GIVE_WAY
                        status = 'ZEBRA-GIVEWAY'
                        v = self._linear * 0.5 * self._traffic_mult * stop_approach_scale * sign_speed_scale
                        self._zebra_active = False
                        self._zebra_phase = 'NONE'
                    else:
                        self._zebra_active = False
                        self._zebra_phase = 'NONE'

            # Mostrar estado
            if self._zebra_active:
                self._reset_steering_filters()
                self._publish_cmd(v, omega)
                self.get_logger().info(f'{status}')
                vis = cv2.resize(zebra_vis, (frame.shape[1], frame.shape[0])) if self._show_window else None
                if vis is not None:
                    self._show_and_maybe_exit(vis, f'{status} {sign_tag} v={v:.3f} w={omega:+.3f}')
                return

        # PRIORIDAD 3: seguidor de línea.
        vis, direction, line_found = find_line(small, previous_direction=self._filtered_dir, roi_y_start=0.50)
        now = self.get_clock().now()

        if line_found:
            dt_dir = (now - self._last_control_time).nanoseconds * 1e-9
            dt_dir = clamp(dt_dir, 0.015, 0.12)

            self._raw_dir_filtered = (1.0 - self._alpha) * self._raw_dir_filtered + self._alpha * direction
            target_dir = self._raw_dir_filtered
            if abs(target_dir) < self._deadband:
                target_dir = 0.0

            max_dir_delta = self._direction_slew_rate * dt_dir
            self._filtered_dir = clamp(target_dir, self._filtered_dir - max_dir_delta, self._filtered_dir + max_dir_delta)
            if abs(self._filtered_dir) < self._deadband:
                self._filtered_dir = 0.0

            self._last_valid_dir = self._filtered_dir
            self._last_line_time = now
            speed_scale = 1.0
            status = 'LINEA'
        else:
            self._filtered_dir = 0.0
            self._prev_filtered_dir = 0.0
            speed_scale = self._lost_scale if not self._stop_when_lost else 0.0
            status = 'SIN-LINEA-RECTO' if not self._stop_when_lost else 'LINEA-PERDIDA-STOP'

        control_now = self.get_clock().now()
        dt = (control_now - self._last_control_time).nanoseconds * 1e-9
        dt = clamp(dt, 0.015, 0.12)
        self._last_control_time = control_now

        d_dir = (self._filtered_dir - self._prev_filtered_dir) / dt
        d_dir = clamp(d_dir, -self._deriv_limit, self._deriv_limit)
        self._prev_filtered_dir = self._filtered_dir

        soft_dir = self._filtered_dir * abs(self._filtered_dir) ** 0.75
        omega_target = -self._kp * soft_dir - self._kd * d_dir
        omega_target = clamp(omega_target, -self._max_w, self._max_w)

        omega_smoothed = (1.0 - self._omega_alpha) * self._omega_filtered + self._omega_alpha * omega_target
        max_delta = self._omega_slew_rate * dt
        omega = clamp(omega_smoothed, self._omega_filtered - max_delta, self._omega_filtered + max_delta)
        omega = clamp(omega, -self._max_w, self._max_w)
        self._omega_filtered = omega

        curve_scale = 1.0 - 0.75 * min(1.0, abs(self._filtered_dir))
        angular_scale = 1.0 - 0.45 * min(1.0, abs(self._omega_filtered) / max(1e-6, self._max_w))
        v = self._linear * speed_scale * stop_approach_scale * self._traffic_mult * sign_speed_scale * curve_scale * angular_scale

        self._publish_cmd(v, omega)
        suffix = f' {sign_tag}' if sign_tag else ''
        if self._pending_turn_valid():
            suffix += f' PENDING-{SIGN_NAMES.get(self._pending_turn, self._pending_turn).upper()}'
        self.get_logger().info(f'{status}{suffix} | dir={self._filtered_dir:+.2f} | v={v:.3f} w={omega:+.3f}')

        if self._show_window:
            vis = cv2.resize(vis, (frame.shape[1], frame.shape[0]))
            self._show_and_maybe_exit(
                vis,
                f'{status}{suffix} dir={self._filtered_dir:+.2f} v={v:.2f} w={omega:+.2f}',
            )

    def destroy_node(self):
        self._publish_stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()