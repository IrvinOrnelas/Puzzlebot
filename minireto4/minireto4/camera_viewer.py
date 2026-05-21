import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import cv2
import numpy as np


WINDOW_TITLE = 'CSI Camera'
MIN_AREA = 10000

# Rangos HSV para rojo (el rojo envuelve los 0/180°, se necesitan dos rangos)
RED_LOWER1 = np.array([0,   120, 70])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([160, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

# Rango HSV para amarillo
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

# Rango HSV para verde
GREEN_LOWER = np.array([40,  50,  50])
GREEN_UPPER = np.array([80, 255, 255])


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    return (
        'nvarguscamerasrc sensor-id=%d ! '
        'video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! '
        'nvvidconv flip-method=%d ! '
        'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
        'videoconvert ! '
        'video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true'
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')

        self.declare_parameter('sensor_id', 0)
        self.declare_parameter('framerate', 30)
        self.declare_parameter('flip_method', 0)

        sensor_id = self.get_parameter('sensor_id').value
        framerate = self.get_parameter('framerate').value
        flip_method = self.get_parameter('flip_method').value

        pipeline = gstreamer_pipeline(
            sensor_id=sensor_id,
            framerate=framerate,
            flip_method=flip_method,
        )
        self.get_logger().info(f'Pipeline: {pipeline}')

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.get_logger().error('No se pudo abrir la cámara CSI')
            return

        self.speed_pub = self.create_publisher(Float32, '/speed_multiplier', 10)

        # Latch de rojo: queda detenido hasta detectar verde
        self._stopped_by_red = False

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
        self.timer = self.create_timer(1.0 / framerate, self.timer_callback)
        self.get_logger().info('Cámara abierta correctamente')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('No se recibió frame')
            return

        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
            self.get_logger().info('Ventana cerrada, apagando nodo...')
            self.cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

        # --- Detección de color ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_red = (
            cv2.inRange(hsv, RED_LOWER1, RED_UPPER1) |
            cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        )
        mask_yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        mask_green  = cv2.inRange(hsv, GREEN_LOWER,  GREEN_UPPER)

        best_contour = None
        best_area = 0
        best_color = None

        color_info = [
            (mask_red,    'rojo',     (0,   0,   255)),
            (mask_yellow, 'amarillo', (0,   255, 255)),
            (mask_green,  'verde',    (0,   255,   0)),
        ]

        # Una sola pasada: detectar y dibujar
        for mask, color_name, color_bgr in color_info:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > MIN_AREA:
                    cv2.drawContours(frame, [cnt], -1, color_bgr, 2)
                    if area > best_area:
                        best_area = area
                        best_contour = cnt
                        best_color = color_name

        # Dibujar el contorno de mayor área con bounding box y publicar velocidad
        speed_msg = Float32()

        color_map = {
            'rojo':     (0,   0,   255),
            'amarillo': (0,   255, 255),
            'verde':    (0,   255,   0),
        }
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            box_color = color_map[best_color]
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
            cv2.putText(
                frame, f'{best_color} ({int(best_area)})',
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2
            )

            if best_color == 'rojo':
                self._stopped_by_red = True
                speed_msg.data = 0.0
            elif best_color == 'verde':
                self._stopped_by_red = False
                speed_msg.data = 1.0
            else:  # amarillo
                speed_msg.data = 0.0 if self._stopped_by_red else 0.5

            self.get_logger().info(
                f'Viendo: {best_color} | area: {int(best_area)} | '
                f'stop_latch: {self._stopped_by_red} | vel: {speed_msg.data}')
        else:
            if self._stopped_by_red:
                speed_msg.data = 0.0
                self.get_logger().info('Viendo: nada (esperando verde) | vel: 0.0')
            else:
                speed_msg.data = 1.0
                self.get_logger().info('Viendo: nada | vel: 1.0')

        self.speed_pub.publish(speed_msg)

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            self.get_logger().info('Tecla q/ESC presionada, apagando nodo...')
            self.cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
