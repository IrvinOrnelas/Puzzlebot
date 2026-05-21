import cv2
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(SCRIPT_DIR, "tomas")

os.makedirs(SAVE_FOLDER, exist_ok=True)


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
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
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


cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("No se pudo abrir la cámara CSI del puzzlebot.")
    exit()

count = 0

print("Presiona 's' para guardar una foto.")
print("Presiona 'f' para terminar el programa.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer imagen de la cámara.")
        break

    cv2.imshow("PuzzleBot - Captura de patron", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = os.path.join(SAVE_FOLDER, f"calib_{count:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Imagen guardada: {filename}")
        count += 1

    elif key == ord("f"):
        break

cap.release()
cv2.destroyAllWindows()
