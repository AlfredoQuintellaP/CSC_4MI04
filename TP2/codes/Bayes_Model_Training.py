import numpy as np
import cv2
import sys

# Classification Bayésienne
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

if len(sys.argv) != 2:
    print("Usage :", sys.argv[0], "<Image_in>")
    sys.exit(2)

img_bgr = cv2.imread(sys.argv[1], -1)
(h_img, w_img, c_img) = img_bgr.shape

print("Dimension de l'image :", h_img, "lignes x", w_img, "colonnes x", c_img, "canaux")
print("Type de l'image :", img_bgr.dtype)

# ================= ROI SELECTION =================

roi_defined = False
r = c = w = h = 0

def define_ROI(event, x, y, flags, param):
    global r, c, w, h, roi_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False

    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y

        w = abs(r2 - r)
        h = abs(c2 - c)

        r = min(r, r2)
        c = min(c, c2)

        roi_defined = True

# ================= MODEL UPDATE =================

def update_GBModel(r, c, w, h, label):
    global img_bgr, clf, data_features, data_labels, first

    # Clamp ROI dentro da imagem
    r_end = min(r + w, w_img)
    c_end = min(c + h, h_img)

    roi_features = img_bgr[c:c_end, r:r_end]

    if roi_features.size == 0:
        print("ROI vide ignorée")
        return

    print("RoI_Features :", roi_features.shape)

    # 1 feature por pixel
    batch_features = roi_features.reshape(-1, 3)
    batch_labels = np.full(batch_features.shape[0], label)

    if first:
        first = False
        data_features = batch_features.copy()
        data_labels = batch_labels.copy()
    else:
        data_features = np.concatenate((data_features, batch_features), axis=0)
        data_labels = np.concatenate((data_labels, batch_labels), axis=0)

# ================= INIT =================

clone = img_bgr.copy()
cv2.namedWindow("Training image")
cv2.setMouseCallback("Training image", define_ROI)

num_pos = 0
num_neg = 0
first = True

# Modelo Bayesiano
clf = QuadraticDiscriminantAnalysis(priors=None)
# clf = GaussianNB(priors=None)

# ================= TRAIN LOOP =================

while True:
    cv2.imshow("Training image", img_bgr)
    key = cv2.waitKey(1) & 0xFF

    if roi_defined:
        cv2.rectangle(img_bgr, (r, c), (r + w, c + h), (0, 255, 0), 2)
    else:
        img_bgr = clone.copy()

    if key == ord("p"):
        num_pos += 1
        update_GBModel(r, c, w, h, 1)
        print("Batch positif n°", num_pos, "enregistré !")

    if key == ord("n"):
        num_neg += 1
        update_GBModel(r, c, w, h, -1)
        print("Batch négatif n°", num_neg, "enregistré !")

    if key == ord("q"):
        break

# ================= TRAIN MODEL =================

print("Dimension des features :", data_features.shape)
print("Dimension des labels :", data_labels.shape)

clf.fit(data_features, data_labels)

# ================= TEST ON IMAGE =================

img_bgr = clone.copy()
img_samples = img_bgr.reshape(-1, 3)

img_labels = clf.predict(img_samples)
img_result = img_labels.reshape(h_img, w_img)

print("Type de l'image de label :", img_result.dtype)

img_result[img_result == 1] = 255
img_result[img_result == -1] = 0
img_result = img_result.astype(np.uint8)

cv2.imshow("Testing on train image", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

