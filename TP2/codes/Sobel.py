import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

# Vérification des arguments
if len(sys.argv) != 2:
    print("Usage :", sys.argv[0], "<Image_in>")
    sys.exit(2)

# Lecture de l'image
img_bgr = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
if img_bgr is None:
    print("Erreur : impossible de lire l'image")
    sys.exit(2)

(h, w, c) = img_bgr.shape
print("Dimension de l'image :", h, "lignes x", w, "colonnes x", c, "canaux")

# Conversion BGR -> RGB pour affichage matplotlib
b, g, r = cv2.split(img_bgr)
img_rgb = cv2.merge([r, g, b])

# Affichage des composantes RGB
plt.figure(figsize=(12,4))
plt.subplot(221)
plt.imshow(img_rgb)
plt.title('Image couleur')
plt.subplot(222)
plt.imshow(r, cmap='gray')
plt.title('Composante Rouge')
plt.subplot(223)
plt.imshow(g, cmap='gray')
plt.title('Composante Verte')
plt.subplot(224)
plt.imshow(b, cmap='gray')
plt.title('Composante Bleue')
plt.show()

# Conversion en HSV
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img_hsv)

# Affichage des composantes HSV
plt.figure(figsize=(12,4))
plt.subplot(221)
plt.imshow(img_rgb)
plt.title('Image couleur')
plt.subplot(222)
plt.imshow(h, cmap='hsv')
plt.title('Composante Teinte')
plt.subplot(223)
plt.imshow(s, cmap='gray')
plt.title('Composante Saturation')
plt.subplot(224)
plt.imshow(v, cmap='gray')
plt.title('Composante Valeur (Luminosité)')
plt.show()

# Conversion en YUV
img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
y, cr, cb = cv2.split(img_yuv)

# Affichage des composantes YUV
plt.figure(figsize=(12,4))
plt.subplot(221)
plt.imshow(img_rgb)
plt.title('Image couleur')
plt.subplot(222)
plt.imshow(y, cmap='gray')
plt.title('Composante Intensité')
plt.subplot(223)
plt.imshow(cr, cmap='gray')
plt.title('Contraste Rouge-Cyan')
plt.subplot(224)
plt.imshow(cb, cmap='gray')
plt.title('Contraste Bleu-Jaune')
plt.show()

# -------------------------------------------------------
# Détection des bordures sur la composante V (Value) du HSV
# -------------------------------------------------------

img_v = np.float64(h)  # Sobel nécessite float64

# Sobel sur X et Y
Ix = cv2.Sobel(img_v, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(img_v, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude du gradient
grad_norm = np.sqrt(Ix**2 + Iy**2)

# Normalisation pour affichage
Ix_disp = cv2.normalize(Ix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
Iy_disp = cv2.normalize(Iy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
grad_disp = cv2.normalize(grad_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Affichage
plt.figure(figsize=(12,4))
plt.subplot(131)
plt.imshow(Ix_disp, cmap='gray')
plt.title("Gradient X (Sobel) sur V")

plt.subplot(132)
plt.imshow(Iy_disp, cmap='gray')
plt.title("Gradient Y (Sobel) sur V")

plt.subplot(133)
plt.imshow(grad_disp, cmap='gray')
plt.title("Magnitude du gradient sur V")
plt.show()

