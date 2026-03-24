import numpy as np
import cv2
import sys
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

if len(sys.argv) != 2:
    print("Usage :", sys.argv[0], "<Image_in>")
    sys.exit(2)

img_bgr = cv2.imread(sys.argv[1], -1)
(h_img, w_img, c_img) = img_bgr.shape

# ================= ESPAÇO DE COR =================
COLOR_SPACE = 'HSV'  # Trocar aqui: 'BGR', 'HSV', 'YCbCr'

def convert_colorspace(img, space):
    if space == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif space == 'YCbCr':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img.copy()

img_converted = convert_colorspace(img_bgr, COLOR_SPACE)

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
# Listas globais — acumulam todas as ROIs corretamente
data_features = []
data_labels = []

def update_GBModel(r, c, w, h, label):
    global data_features, data_labels
    r_end = min(r + w, w_img)
    c_end = min(c + h, h_img)
    roi_features = img_converted[c:c_end, r:r_end]
    if roi_features.size == 0:
        print("ROI vide ignorée")
        return
    batch_features = roi_features.reshape(-1, 3).astype(np.float64)
    batch_labels = np.full(batch_features.shape[0], label)
    # append nas listas — sem problema de first/global
    data_features.append(batch_features)
    data_labels.append(batch_labels)
    print(f"  ROI ajoutée : {batch_features.shape[0]} pixels, label={label}")

# ================= INIT =================
clone_bgr = img_bgr.copy()
cv2.namedWindow("Training image")
cv2.setMouseCallback("Training image", define_ROI)
num_pos = num_neg = 0

while True:
    cv2.imshow("Training image", img_bgr)
    key = cv2.waitKey(1) & 0xFF
    if roi_defined:
        cv2.rectangle(img_bgr, (r, c), (r + w, c + h), (0, 255, 0), 2)
    else:
        img_bgr = clone_bgr.copy()
    if key == ord("p"):
        num_pos += 1
        update_GBModel(r, c, w, h, 1)
        print(f"Batch positif n°{num_pos} enregistré !")
    if key == ord("n"):
        num_neg += 1
        update_GBModel(r, c, w, h, -1)
        print(f"Batch négatif n°{num_neg} enregistré !")
    if key == ord("q"):
        break

# ================= TREINAR E COMPARAR =================
# Concatenar todas as ROIs acumuladas
X = np.concatenate(data_features, axis=0)
y = np.concatenate(data_labels,   axis=0)
print(f"Total pixels de treino : {X.shape[0]} ({np.sum(y==1)} positifs, {np.sum(y==-1)} négatifs)")

img_samples = img_converted.reshape(-1, 3).astype(np.float64)

def show_result(clf, name):
    clf.fit(X, y)
    img_labels = clf.predict(img_samples).reshape(h_img, w_img)
    img_result = np.where(img_labels == 1, 255, 0).astype(np.uint8)
    cv2.imshow(f"{name} - {COLOR_SPACE}", img_result)
    fname = f"bayes_{name}_{COLOR_SPACE}.png"
    cv2.imwrite(fname, img_result)
    print(f"Saved: {fname}")

show_result(QuadraticDiscriminantAnalysis(priors=None), "QDA")
show_result(GaussianNB(priors=None),                    "GaussianNB")

cv2.waitKey(0)
cv2.destroyAllWindows()
