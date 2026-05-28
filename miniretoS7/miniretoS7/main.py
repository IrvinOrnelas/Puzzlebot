import argparse
import time
import cv2

from actividad_2_06 import TrafficSignDetection


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    """Pipeline GStreamer para la cámara CSI del Puzzlebot (Jetson)."""
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


def run(sensor_id=0, window_name="Traffic Sign Detection — Puzzlebot"):
    pipeline = gstreamer_pipeline(sensor_id=sensor_id)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara CSI del Puzzlebot.")

    detector = TrafficSignDetection()

    try:
        print("Cámara CSI abierta. Presiona 'q' para salir.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            annotated, front_sign = detector.process(frame)
            fps = 1.0 / (time.time() - t0 + 1e-6)

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prueba de detección de señales con cámara CSI del Puzzlebot. Presiona 'q' para salir."
    )
    parser.add_argument("--sensor", "-s", type=int, default=0, help="ID del sensor CSI (default: 0)")
    args = parser.parse_args()

    run(args.sensor)
