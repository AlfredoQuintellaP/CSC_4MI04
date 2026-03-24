import numpy as np
import cv2
import sys
from sklearn.cluster import KMeans

if len(sys.argv) < 2:
    print("Usage:", sys.argv[0], "<Image_train> [Image_test]")
    sys.exit(2)

# ================= CHARGEMENT =================
img_bgr = cv2.imread(sys.argv[1], -1)
(h_img, w_img, c) = img_bgr.shape
print("Dimension de l'image :", h_img, "lignes x", w_img, "colonnes x", c, "canaux")

# ================= ESPACE DE COULEUR =================
# Choisir ici : 'BGR', 'HSV', 'YCbCr', 'LAB'
COLOR_SPACE = 'BGR'

def convert_colorspace(img_bgr, space):
    if space == 'HSV':
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    elif space == 'YCbCr':
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    elif space == 'LAB':
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    else:  # BGR
        return img_bgr.copy()

img_converted = convert_colorspace(img_bgr, COLOR_SPACE)

# ================= ENTRAÎNEMENT K-MEANS =================
Nb_classes = 6  # Valeur optimale trouvée expérimentalement
img_samples = np.reshape(img_converted, (-1, 3)).astype(np.float32)

kmeans = KMeans(n_clusters=Nb_classes, random_state=0, n_init=10)
kmeans.fit(img_samples)

print("Centres des clusters :", kmeans.cluster_centers_)

# ================= AFFICHAGE SUR IMAGE D'ENTRAÎNEMENT =================
img_labels = np.reshape(kmeans.labels_, (h_img, w_img))
img_labels_display = (img_labels * 255 / (Nb_classes - 1)).astype(np.uint8)
cv2.imshow(f"Clusters - image train ({COLOR_SPACE}, K={Nb_classes})", img_labels_display)
cv2.waitKey(0)

# ================= PRÉDICTION SUR NOUVELLE IMAGE =================
if len(sys.argv) == 3:
    img_test_bgr = cv2.imread(sys.argv[2], -1)
    if img_test_bgr is None:
        print("Erreur : image de test introuvable.")
        sys.exit(1)

    (h_test, w_test, _) = img_test_bgr.shape
    img_test_converted = convert_colorspace(img_test_bgr, COLOR_SPACE)

    # predict() applique le modèle appris sur une nouvelle image
    new_img_samples = np.reshape(img_test_converted, (-1, 3)).astype(np.float32)
    new_labels = kmeans.predict(new_img_samples)

    img_test_labels = np.reshape(new_labels, (h_test, w_test))
    img_test_display = (img_test_labels * 255 / (Nb_classes - 1)).astype(np.uint8)

    cv2.imshow(f"Clusters - image test ({COLOR_SPACE}, K={Nb_classes})", img_test_display)
    cv2.waitKey(0)

cv2.destroyAllWindows()
