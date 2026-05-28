"""
ROS 2 node: sign_detector_node
Detecta señales de tráfico usando YOLOv10 y la cámara CSI del Puzzlebot (Jetson).

Topics publicados:
  /sign_detected  (std_msgs/String)  — nombre de la señal frente al robot, o vacío.
  /sign_image     (sensor_msgs/Image) — frame anotado con las detecciones.
"""

import threading

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image

from miniretoS7.actividad_2_06 import TrafficSignDetection


# ---------------------------------------------------------------------------
# GStreamer / CSI camera helpers (Jetson Nano / Orin)
# ---------------------------------------------------------------------------

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
        'nvarguscamerasrc sensor-id=%d do-timestamp=true tnr-mode=0 ee-mode=0 ! '
        'video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! '
        'queue max-size-buffers=1 leaky=downstream ! '
        'nvvidconv flip-method=%d ! '
        'video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12 ! '
        'queue max-size-buffers=1 leaky=downstream ! '
        'nvvidconv ! '
        'video/x-raw, format=(string)BGRx ! '
        'videoconvert ! '
        'video/x-raw, format=(string)BGR ! '
        'appsink max-buffers=1 drop=true sync=false emit-signals=false'
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


class CSICamera:
    """Captura en hilo dedicado para siempre tener el frame más reciente."""

    def __init__(
        self,
        sensor_id=0,
        capture_width=1280,
        capture_height=720,
        display_width=640,
        display_height=360,
        framerate=30,
        flip_method=0,
        logger=None,
    ):
        self.logger = logger
        pipeline = gstreamer_pipeline(
            sensor_id=sensor_id,
            capture_width=capture_width,
            capture_height=capture_height,
            display_width=display_width,
            display_height=display_height,
            framerate=framerate,
            flip_method=flip_method,
        )
        if self.logger:
            self.logger.info(f'Pipeline: {pipeline}')

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError('No se pudo abrir la cámara CSI del Puzzlebot.')

        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self):
        with self._lock:
            return self._ret, (self._frame.copy() if self._frame is not None else None)

    def release(self):
        self._running = False
        self._thread.join(timeout=2.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

class SignDetectorNode(Node):

    def __init__(self):
        super().__init__('sign_detector_node')

        # Parámetros de cámara
        self.declare_parameter('sensor_id', 0)
        self.declare_parameter('capture_width', 1280)
        self.declare_parameter('capture_height', 720)
        self.declare_parameter('display_width', 640)
        self.declare_parameter('display_height', 360)
        self.declare_parameter('framerate', 30)
        self.declare_parameter('flip_method', 0)

        # Parámetros de detección
        self.declare_parameter('blur_threshold', 100.0)
        self.declare_parameter('show_window', False)

        framerate = int(self.get_parameter('framerate').value)

        # Cámara CSI
        self.camera = CSICamera(
            sensor_id=int(self.get_parameter('sensor_id').value),
            capture_width=int(self.get_parameter('capture_width').value),
            capture_height=int(self.get_parameter('capture_height').value),
            display_width=int(self.get_parameter('display_width').value),
            display_height=int(self.get_parameter('display_height').value),
            framerate=framerate,
            flip_method=int(self.get_parameter('flip_method').value),
            logger=self.get_logger(),
        )

        # Detector YOLO
        self.detector = TrafficSignDetection(
            blur_threshold=float(self.get_parameter('blur_threshold').value)
        )

        self.show_window = bool(self.get_parameter('show_window').value)
        self.bridge = CvBridge()

        # Publishers
        self.pub_sign = self.create_publisher(String, '/sign_detected', 10)
        self.pub_image = self.create_publisher(Image, '/sign_image', 10)

        self.timer = self.create_timer(1.0 / framerate, self.timer_callback)
        self.get_logger().info('SignDetectorNode iniciado — cámara CSI del Puzzlebot.')

    def timer_callback(self):
        ret, frame = self.camera.read()
        if not ret or frame is None:
            self.get_logger().warn('No se recibió frame de la cámara CSI.')
            return

        annotated, front_sign = self.detector.process(frame)

        # Publicar señal detectada
        msg_sign = String()
        msg_sign.data = front_sign if front_sign is not None else ''
        self.pub_sign.publish(msg_sign)

        # Publicar imagen anotada
        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_image.publish(img_msg)

        if self.show_window:
            cv2.imshow('Sign Detector — Puzzlebot', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()

    def destroy_node(self):
        self.camera.release()
        cv2.destroyAllWindows()
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = SignDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
