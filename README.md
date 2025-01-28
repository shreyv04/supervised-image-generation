# supervised-image-generation
Supervised Image Generation - GANs and CGANs

## Introduction
This project explores the implementation of Generative Adversarial Networks (GANs) and Conditional GANs (CGANs) for supervised image generation tasks. Leveraging Deep Convolutional GANs (DCGANs), we focus on synthesizing high-quality images using two distinct datasets: MNIST and CelebA. The project emphasizes architectural innovation, data preprocessing, and optimization strategies to overcome challenges like training instability and mode collapse.

## Overview
- **Datasets**:  
  - **MNIST**: Grayscale handwritten digits (28x28 resolution).  
  - **CelebA**: RGB celebrity face images (resized to 32x32).  

- **Architecture**:  
  - Utilizes DCGANs with convolutional layers, batch normalization, and Leaky ReLU/Tanh activations for stable training.  
  - Incorporates label conditioning in CGANs for controlled image generation.

- **Evaluation**:  
  - Visual quality of generated images.  
  - Consistency of outputs for labeled data in CGANs.  

## Key Features
- **Multi-dataset Compatibility**:  
  Scalable architectures designed for both simple (MNIST) and complex (CelebA) datasets.  

- **Advanced Techniques**:  
  Integration of transposed convolutions, conditional embeddings, and binary cross-entropy loss for high-quality outputs.  

- **Customizable**:  
  Hyperparameters and architectural settings optimized for diverse data complexities.  
