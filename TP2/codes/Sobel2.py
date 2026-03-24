import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

# -----------------------------
# Vérification des arguments
# -----------------------------
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

# -----------------------------
# Conversion BGR -> RGB pour matplotlib
# -----------------------------
b, g, r = cv2.split(img_bgr)
img_rgb = cv2.merge([r, g, b])

# -----------------------------
# Conversion HSV et extraction du canal V
# -----------------------------
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h_channel, s_channel, v_channel = cv2.split(img_hsv)

# -----------------------------
# Application de CLAHE sur le canal V
# -----------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
v_clahe = clahe.apply(v_channel)

# -----------------------------
# Fonction pour calculer le gradient Sobel
# -----------------------------
def calculer_gradient_sobel(image):
    # Convertir en float64 pour le calcul
    img_float = image.astype(np.float64)
    
    # Calcul des gradients horizontaux et verticaux
    Ix = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude du gradient
    gradient = np.sqrt(Ix**2 + Iy**2)
    
    # Normalisation pour l'affichage (0-255)
    gradient_normalise = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return gradient_normalise

# -----------------------------
# Calcul des gradients
# -----------------------------
# Gradient de l'image originale (convertie en niveaux de gris)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
gradient_original = calculer_gradient_sobel(img_gray)

# Gradient du canal V original
gradient_v_original = calculer_gradient_sobel(v_channel)

# Gradient du canal V après CLAHE
gradient_v_clahe = calculer_gradient_sobel(v_clahe)

# -----------------------------
# Premier affichage: Images
# -----------------------------
plt.figure()
plt.subplot(221)
plt.imshow(img_rgb)
plt.title('Image couleur', fontsize=12)
plt.axis('off')

plt.subplot(222)
plt.imshow(v_channel, cmap='gray')
plt.title('Canal V original', fontsize=12)
plt.axis('off')

plt.subplot(223)
plt.imshow(v_clahe, cmap='gray')
plt.title('Canal V après CLAHE', fontsize=12)
plt.axis('off')

plt.subplot(224)
# Espace vide ou on pourrait mettre une autre information
plt.axis('off')
plt.tight_layout()
plt.show()

# -----------------------------
# Deuxième affichage: Gradients Sobel
# -----------------------------
plt.figure()
plt.subplot(221)
plt.imshow(img_rgb)
plt.title('Image couleur', fontsize=12)
plt.axis('off')

plt.subplot(222)
plt.imshow(gradient_original, cmap='gray')
plt.title('Sobel - Image originale (gris)', fontsize=12)
plt.axis('off')

plt.subplot(223)
plt.imshow(gradient_v_original, cmap='gray')
plt.title('Sobel - Canal V original', fontsize=12)
plt.axis('off')

plt.subplot(224)
plt.imshow(gradient_v_clahe, cmap='gray')
plt.title('Sobel - Canal V après CLAHE', fontsize=12)
plt.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------
# Troisième affichage: Comparaison complète (6 subplots)
# -----------------------------
plt.figure(figsize=(15, 10))

# Ligne 1: Canaux V
plt.subplot(2, 3, 1)
plt.imshow(v_channel, cmap='gray')
plt.title('Canal V original', fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(v_clahe, cmap='gray')
plt.title('Canal V après CLAHE', fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.axis('off')  # Espace vide

# Ligne 2: Gradients Sobel
plt.subplot(2, 3, 4)
plt.imshow(gradient_original, cmap='gray')
plt.title('Sobel - Image originale (gris)', fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(gradient_v_original, cmap='gray')
plt.title('Sobel - Canal V original', fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(gradient_v_clahe, cmap='gray')
plt.title('Sobel - Canal V après CLAHE', fontsize=12)
plt.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------
# Quatrième affichage: Alternative - Avec image couleur comme référence
# -----------------------------
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Image couleur', fontsize=12)

plt.subplot(2, 3, 2)
plt.imshow(v_channel, cmap='gray')
plt.title('Canal V original', fontsize=12)

plt.subplot(2, 3, 3)
plt.imshow(v_clahe, cmap='gray')
plt.title('Canal V après CLAHE', fontsize=12)

# Ligne 2: Gradients Sobel
plt.subplot(2, 3, 4)
plt.imshow(gradient_original, cmap='gray')
plt.title('Sobel - Image originale (gris)', fontsize=12)

plt.subplot(2, 3, 5)
plt.imshow(gradient_v_original, cmap='gray')
plt.title('Sobel - Canal V original', fontsize=12)

plt.subplot(2, 3, 6)
plt.imshow(gradient_v_clahe, cmap='gray')
plt.title('Sobel - Canal V après CLAHE', fontsize=12)

plt.tight_layout()
plt.show()

# -----------------------------
# Statistiques
# -----------------------------
print("\n--- Statistiques des gradients ---")
print(f"Gradient original (gris): min={gradient_original.min()}, max={gradient_original.max()}, mean={gradient_original.mean():.2f}")
print(f"Gradient V original: min={gradient_v_original.min()}, max={gradient_v_original.max()}, mean={gradient_v_original.mean():.2f}")
print(f"Gradient V CLAHE: min={gradient_v_clahe.min()}, max={gradient_v_clahe.max()}, mean={gradient_v_clahe.mean():.2f}")
