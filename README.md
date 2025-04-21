# Adversarial-Attack-on-Face-Recognition-Model

Overview

This project explores adversarial attacks on deep learning-based face recognition models. Using the FaceNet model from facenet-pytorch, we implement and evaluate adversarial perturbations using Fast Gradient Sign Method (FGSM) and DeepFool Attack from foolbox. The impact of these attacks is analyzed by comparing face embeddings before and after adversarial perturbation.

Features

-Face Recognition Model: Utilizes facenet-pytorch for feature extraction.

-Adversarial Attacks: Implements FGSM, DeepFool, PGD, BIM attacks using foolbox.

-Embedding Similarity Analysis: Measures the impact of attacks using cosine similarity,L2 distance,SSIM,PSNR and Perturbation method between original and adversarial embeddings.

-Visualization: Displays perturbed images to highlight the changes introduced by adversarial attacks.

Results

-The adversarial examples, except Deepfool successfully fooled the FaceNet model, reducing similarity scores.


