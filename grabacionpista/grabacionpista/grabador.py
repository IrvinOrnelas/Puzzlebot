import os
from datetime import datetime

import cv2
import rclpy
from rclpy.node import Node


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


class GrabadorPista(Node):

    DISPLAY_W = 640
    DISPLAY_H = 360
    FPS = 30

    def __init__(self):
        super().__init__('grabador_pista')

        self.cap = cv2.VideoCapture(
            gstreamer_pipeline(
                display_width=self.DISPLAY_W,
                display_height=self.DISPLAY_H,
                framerate=self.FPS,
            ),
            cv2.CAP_GSTREAMER,
        )

        if not self.cap.isOpened():
            self.get_logger().error('No se pudo abrir la camara CSI.')
            raise RuntimeError('No se pudo abrir la camara CSI.')

        self._recording = False
        self._writer = None
        self._output_path = None

        self.get_logger().info(
            'Camara lista.\n'
            '  [x] o [d] -> Iniciar grabacion\n'
            '  [f]       -> Detener grabacion\n'
            '  [q]       -> Salir'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_output_path(self) -> str:
        videos_dir = '/home/puzzlebot/ros2_ws/src/grabacionpista/videos'
        os.makedirs(videos_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(videos_dir, f'grabacion_{timestamp}.mp4')

    def _start_recording(self, frame):
        self._output_path = self._new_output_path()
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(self._output_path, fourcc, float(self.FPS), (w, h))
        self._recording = True
        self.get_logger().info(f'Grabando -> {self._output_path}')

    def _stop_recording(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._recording = False
        self.get_logger().info(f'Grabacion guardada: {self._output_path}')

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warning('No se pudo leer un frame. Reintentando...')
                continue

            # Escribir frame si se esta grabando
            if self._recording:
                self._writer.write(frame)
                cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)          # punto rojo
                cv2.putText(frame, 'REC', (38, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'Presiona X o D para grabar',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            cv2.imshow('Grabacion Pista', frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('x'), ord('d')):
                if not self._recording:
                    self._start_recording(frame)
                else:
                    self.get_logger().info('Ya se esta grabando. Presiona [f] para detener.')

            elif key == ord('f'):
                if self._recording:
                    self._stop_recording()
                else:
                    self.get_logger().info('No hay grabacion activa.')

            elif key == ord('q'):
                break

            rclpy.spin_once(self, timeout_sec=0)

        # Cleanup
        if self._recording:
            self._stop_recording()
        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = GrabadorPista()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
