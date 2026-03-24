# Image Recognition (Reconnaissance d'Image)

This repository contains the practical work carried out for the **Image Recognition** course, which focuses on fundamental techniques for image processing and analysis.
The course is essentially structured around **three practical projects (TPs)**, each exploring a different aspect of image recognition and analysis.

## TP1 - Détection et Appariement de Points Caractéristiques 

**Objective:**

This practical focuses on detecting feature points in images and matching them across image pairs to evaluate the robustness and accuracy of different detectors and descriptors.

**Tools and methods used:**

- Python & OpenCV: for image reading, keypoint detection, and descriptor computation.

- Detectors and descriptors:

    - ORB: a fast binary detector and descriptor, rotation invariant and partially scale invariant.

    - KAZE: a floating-point detector and descriptor based on anisotropic diffusion, robust to scale variations and intensity changes.

- Matching methods:

    - Ratio Test (Lowe): to filter ambiguous matches.

    - Can use BFMatcher or FLANN depending on the type of descriptor.

- Quantitative evaluation:

    - Applying a known geometric transformation (cv2.warpAffine) to generate a transformed image.

    - Calculating the Euclidean error between the detected points and their theoretical positions to measure matching accuracy.

    - Optionally visualizing errors using a color map to intuitively highlight regions of high deviation.

## TP2 - Classification de caractéristiques locales Approches supervisée (bayésienne) et non supervisée (K-Means)

**Objective:**
Segmentation of road scenes from the KITTI dataset by classifying pixels into road / non-road, using supervised and unsupervised approaches.

**Tools and methods used:**
- Python, OpenCV & Scikit-Learn.
- Supervised approach — Bayesian classification:
    - Manual ROI annotation to train the model.
    - QDA (Quadratic Discriminant Analysis): full covariance matrix per class, captures inter-channel correlations.
    - GaussianNB: assumes channel independence; performs comparably to QDA in YCbCr space.
    - Best result: GaussianNB in YCbCr color space.
- Unsupervised approach — K-Means clustering:
    - Tested K ∈ {4, 6, 8, 10} and color spaces BGR, HSV, YCbCr.
    - Best result: K=8 in BGR space, with cross-image generalization via `predict()`.
- Color spaces evaluated: BGR, HSV, YCbCr — YCbCr most robust to lighting variations for Bayesian classification; BGR most effective for K-Means due to chromatic homogeneity of roads in KITTI.

## TP3
