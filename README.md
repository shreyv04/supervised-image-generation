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

## Results
 - **GANs using MNIST Dataset**
   <img width="1092" alt="Figure 2025-01-09 172012 (79)" src="https://github.com/user-attachments/assets/3c220073-14d3-4e32-971e-00fb6a4adae2" />

- **GANs using CelebA Dataset**
  ![gans-2](https://github.com/user-attachments/assets/8bb3403e-ddf7-41d2-ab93-cdb49839b909)

- **CGANs using MNIST Dataset**
  ![gans-2](https://github.com/user-attachments/assets/ca15014d-1258-4df2-8939-42b1b0a23214)

- **CGANs using CelebA Dataset**
  ![cgan-3](https://github.com/user-attachments/assets/981ecdb6-39d4-449c-b12a-cc2485528b96)
