# Adversarial-Attack-on-Face-Recognition-Model

Overview

This project explores adversarial attacks on deep learning-based face recognition models. Using the FaceNet model from facenet-pytorch, we implement and evaluate adversarial perturbations using Fast Gradient Sign Method (FGSM) and DeepFool Attack from foolbox. The impact of these attacks is analyzed by comparing face embeddings before and after adversarial perturbation.

Features

-Face Recognition Model: Utilizes facenet-pytorch for feature extraction.

-Adversarial Attacks: Implements FGSM and DeepFool attacks using foolbox.

-Embedding Similarity Analysis: Measures the impact of attacks using cosine similarity between original and adversarial embeddings.

-Visualization: Displays perturbed images to highlight the changes introduced by adversarial attacks.

Results

-The adversarial examples successfully fooled the FaceNet model, reducing cosine similarity scores.

-FGSM resulted in small perturbations but was noticeable in high-confidence cases.

-DeepFool generated more optimized perturbations with a lower attack budget.
