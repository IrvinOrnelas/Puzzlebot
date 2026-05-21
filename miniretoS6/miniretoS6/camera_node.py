import cv2
import rclpy
from rclpy.node import Node


WINDOW_TITLE = 'CSI Camera'


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    """Pipeline para cámara CSI en Jetson usando nvarguscamerasrc."""
    return (
        'nvarguscamerasrc sensor-id=%d ! '
        'video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! '
        'nvvidconv flip-method=%d ! '
        'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
        'videoconvert ! '
        'video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true sync=false'
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
    """
    Clase reutilizable para abrir la cámara una sola vez.

    line_follower importa esta clase para no ejecutar dos nodos que se peleen
    por la misma cámara CSI.
    """

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
        self.pipeline = gstreamer_pipeline(
            sensor_id=sensor_id,
            capture_width=capture_width,
            capture_height=capture_height,
            display_width=display_width,
            display_height=display_height,
            framerate=framerate,
            flip_method=flip_method,
        )
        if self.logger:
            self.logger.info(f'Pipeline: {self.pipeline}')

        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError('No se pudo abrir la cámara CSI')

    def read(self):
        return self.cap.read()

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


class CameraNode(Node):
    """Nodo de prueba: solo abre la cámara y muestra la imagen."""

    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('sensor_id', 0)
        self.declare_parameter('framerate', 30)
        self.declare_parameter('flip_method', 0)
        self.declare_parameter('capture_width', 1280)
        self.declare_parameter('capture_height', 720)
        self.declare_parameter('display_width', 640)
        self.declare_parameter('display_height', 360)

        self.framerate = int(self.get_parameter('framerate').value)

        self.camera = CSICamera(
            sensor_id=int(self.get_parameter('sensor_id').value),
            capture_width=int(self.get_parameter('capture_width').value),
            capture_height=int(self.get_parameter('capture_height').value),
            display_width=int(self.get_parameter('display_width').value),
            display_height=int(self.get_parameter('display_height').value),
            framerate=self.framerate,
            flip_method=int(self.get_parameter('flip_method').value),
            logger=self.get_logger(),
        )

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
        self.timer = self.create_timer(1.0 / self.framerate, self.timer_callback)
        self.get_logger().info('Cámara abierta correctamente')

    def timer_callback(self):
        ret, frame = self.camera.read()
        if not ret:
            self.get_logger().warn('No se recibió frame')
            return

        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
            raise SystemExit

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            raise SystemExit

    def destroy_node(self):
        self.camera.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

