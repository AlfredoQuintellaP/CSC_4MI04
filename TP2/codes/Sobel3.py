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

# Conversion BGR -> RGB pour matplotlib
b, g, r = cv2.split(img_bgr)
img_rgb = cv2.merge([r, g, b])

# -----------------------------
# Conversion HSV
# -----------------------------
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h_channel, s_channel, v_channel = cv2.split(img_hsv)

# -----------------------------
# Étape 1: Masque des pixels rouges (large)
# -----------------------------
lower_red1 = np.array([0, 30, 30])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 30, 30])
upper_red2 = np.array([179, 255, 255])

mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
mask_red_large = cv2.bitwise_or(mask_red1, mask_red2)

# -----------------------------
# Étape 2: Filtrer pour garder seulement les pixels rouges 
#          qui ont des voisins rouges
# -----------------------------
# Définir un noyau pour la convolution (voisinage)
kernel = np.ones((5,5), np.uint8)

# Compter le nombre de pixels rouges dans chaque voisinage 5x5
red_density = cv2.filter2D(mask_red_large.astype(np.float32), -1, kernel)

# Garder seulement les pixels où il y a au moins X pixels rouges dans le voisinage
min_red_neighbors = 12  # Ajustez ce seuil (0-25 pour noyau 5x5)
mask_red_dense = (red_density >= min_red_neighbors).astype(np.uint8) * 255

# Optionnel: Opérations morphologiques pour nettoyer le masque
mask_red_dense = cv2.morphologyEx(mask_red_dense, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_red_dense = cv2.morphologyEx(mask_red_dense, cv2.MORPH_OPEN, kernel, iterations=1)

# -----------------------------
# Étape 3: Alternative - Détection de composants connexes
# -----------------------------
# Trouver les composants connexes (régions)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_red_large, connectivity=8)

# Créer un masque avec seulement les régions d'une taille minimale
min_region_size = 100  # Ajustez ce seuil (en pixels)
mask_red_components = np.zeros_like(mask_red_large)

for i in range(1, num_labels):  # Ignorer l'arrière-plan (0)
    if stats[i, cv2.CC_STAT_AREA] >= min_region_size:
        mask_red_components[labels == i] = 255

# -----------------------------
# Étape 4: Remap uniquement pour les régions rouges denses
# -----------------------------
# Choisissez une méthode: mask_red_dense OU mask_red_components
mask_red_final = mask_red_components  # Je recommande celle-ci

h_remap = h_channel.copy().astype(np.float32)

# Pour les pixels rouges denses (plage 0-10)
mask_red1_bool = (h_channel >= 0) & (h_channel <= 10) & (mask_red_final > 0)
h_remap[mask_red1_bool] = 120 - (10 - h_remap[mask_red1_bool])

# Pour les pixels rouges denses (plage 160-179)
mask_red2_bool = (h_channel >= 160) & (h_channel <= 179) & (mask_red_final > 0)
h_remap[mask_red2_bool] = 120 + (h_remap[mask_red2_bool] - 170)

h_remap = np.clip(h_remap, 0, 179).astype(np.uint8)

# -----------------------------
# Recomposer image HSV remapée
# -----------------------------
img_hsv_remap = cv2.merge([h_remap, s_channel, v_channel])
img_rgb_remap = cv2.cvtColor(img_hsv_remap, cv2.COLOR_HSV2RGB)

# -----------------------------
# CLAHE sur V
# -----------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
v_eq = clahe.apply(v_channel)

# -----------------------------
# Gradients
# -----------------------------
Ix_v = cv2.Sobel(v_eq.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
Iy_v = cv2.Sobel(v_eq.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
grad_v = np.sqrt(Ix_v**2 + Iy_v**2)
grad_v_disp = cv2.normalize(grad_v, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

h_remap_float = h_remap.astype(np.float64)
Ix_h = cv2.Sobel(h_remap_float, cv2.CV_64F, 1, 0, ksize=3)
Iy_h = cv2.Sobel(h_remap_float, cv2.CV_64F, 0, 1, ksize=3)
grad_h = np.sqrt(Ix_h**2 + Iy_h**2)
grad_h_disp = cv2.normalize(grad_h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

grad_comb_disp = cv2.addWeighted(grad_v_disp, 0.7, grad_h_disp, 0.3, 0)

# -----------------------------
# Visualisation améliorée
# -----------------------------
plt.figure(figsize=(18,12))

plt.subplot(341)
plt.imshow(img_rgb)
plt.title("Image originale")
plt.axis('off')

plt.subplot(342)
plt.imshow(mask_red_large, cmap='gray')
plt.title("Masque rouge (large)")
plt.axis('off')

plt.subplot(343)
plt.imshow(red_density, cmap='hot')
plt.title("Densité de pixels rouges")
plt.colorbar()
plt.axis('off')

plt.subplot(344)
plt.imshow(mask_red_dense, cmap='gray')
plt.title(f"Rouge dense (min {min_red_neighbors} voisins)")
plt.axis('off')

plt.subplot(345)
plt.imshow(mask_red_components, cmap='gray')
plt.title(f"Composants rouges (min {min_region_size} px)")
plt.axis('off')

plt.subplot(346)
plt.imshow(img_rgb_remap)
plt.title("Image avec rouge→bleu (régions)")
plt.axis('off')

plt.subplot(347)
plt.imshow(h_channel, cmap='hsv')
plt.title("Hue originale")
plt.axis('off')

plt.subplot(348)
plt.imshow(h_remap, cmap='hsv')
plt.title("Hue après remap")
plt.axis('off')

plt.subplot(349)
plt.imshow(v_eq, cmap='gray')
plt.title("V CLAHE")
plt.axis('off')

plt.subplot(3,4,10)
plt.imshow(grad_v_disp, cmap='gray')
plt.title("Gradient V")
plt.axis('off')

plt.subplot(3,4,11)
plt.imshow(grad_h_disp, cmap='gray')
plt.title("Gradient Hue")
plt.axis('off')

plt.subplot(3,4,12)
plt.imshow(grad_comb_disp, cmap='gray')
plt.title("Gradient combiné")
plt.axis('off')

plt.tight_layout()
plt.show()
