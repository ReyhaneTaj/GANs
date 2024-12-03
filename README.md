GAN Projects
A collection of Generative Adversarial Network (GAN) projects for various applications, including fraud detection, image generation, and more.

Projects
1. Fraud Detection with GANs
Code and Notebook
Dataset: Kaggle Credit Card Fraud Detection Dataset
Overview:
This project leverages GANs to generate synthetic data for the minority class (fraudulent transactions), addressing class imbalance and improving the performance of fraud detection models.
2. GANDataBalancer
The GANDataBalancer class is a flexible tool designed to handle imbalanced datasets effectively. It:

Automatically identifies the minority and majority classes in a dataset.
Trains a GAN to generate high-quality synthetic data for the minority class.
Combines synthetic and original data to create a balanced dataset.
Provides a single-step method (fit_resample) to train the GAN and generate balanced data effortlessly.
This class is especially useful for tasks like fraud detection, where class imbalance poses significant challenges for machine learning models.
