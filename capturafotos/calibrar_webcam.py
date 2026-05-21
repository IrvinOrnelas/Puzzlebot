import cv2
import numpy as np
import glob
import os

# ==========================
# CONFIGURACIÓN
# ==========================

# El patrón del PDF tiene 6 x 8 cuadros,
# por lo tanto tiene 5 x 7 esquinas internas.
CHECKERBOARD = (5, 7)

# Tamaño real de cada cuadro: 3 cm = 0.03 m
SQUARE_SIZE = 0.03

CALIBRATION_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tomas")

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ==========================
# PUNTOS 3D DEL PATRÓN
# ==========================

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

images = glob.glob(os.path.join(CALIBRATION_FOLDER, "*.jpg"))

if len(images) == 0:
    print("No hay imágenes en la carpeta tomas.")
    print("Primero ejecuta capturar_webcam.py y guarda fotos con 's'.")
    exit()

print(f"Imágenes encontradas: {len(images)}")

gray = None
valid_images = 0

for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print("No se pudo leer:", fname)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        valid_images += 1
        objpoints.append(objp)

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        imgpoints.append(corners_refined)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
        cv2.imshow("Esquinas detectadas", img)
        cv2.waitKey(300)

        print("Detectado:", fname)

    else:
        print("No detectado:", fname)

cv2.destroyAllWindows()

print(f"\nImágenes válidas para calibración: {valid_images}")

if valid_images < 8:
    print("Muy pocas imágenes válidas.")
    print("Toma más fotos donde el patrón se vea completo, enfocado y bien iluminado.")
    exit()

# ==========================
# CALIBRACIÓN
# ==========================

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("\n========== RESULTADOS ==========")

print("\nError RMS:")
print(ret)

print("\nMatriz K:")
print(K)

print("\nCoeficientes de distorsión:")
print(dist)

k1 = dist[0][0]
k2 = dist[0][1]

print("\nk1 =", k1)
print("k2 =", k2)

np.savez("parametros_webcam.npz", K=K, dist=dist)

print("\nParámetros guardados en parametros_webcam.npz")

# ==========================
# CORRECCIÓN DE DISTORSIÓN
# Toma una imagen arbitraria, remueve la distorsión y la muestra en pantalla.
# ==========================

test_image_path = images[0]
test_img = cv2.imread(test_image_path)

if test_img is None:
    print("No se pudo leer la imagen de prueba:", test_image_path)
    exit()

print(f"\nImagen de prueba: {test_image_path}")

h, w = test_img.shape[:2]

new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

undistorted = cv2.undistort(test_img, K, dist, None, new_K)

cv2.imshow("Original", test_img)
cv2.imshow("Sin distorsion", undistorted)

print("Presiona cualquier tecla para cerrar.")
cv2.waitKey(0)
cv2.destroyAllWindows()
