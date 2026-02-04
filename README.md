# Image Recognition (Reconnaissance d'Image)

This repository contains the practical work carried out for the **Image Recognition** course, which focuses on fundamental techniques for image processing and analysis.
The course is essentially structured around **three practical projects (TPs)**, each exploring a different aspect of image recognition and analysis.

## TP1 - Detection and Matching of Feature Points

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

## TP2

## TP3
