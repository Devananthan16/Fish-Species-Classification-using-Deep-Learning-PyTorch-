# Fish-Species-Classification-using-Deep-Learning-PyTorch-
This project is a Multiclass Fish Image Classifier built using PyTorch.   It identifies different fish species from images using deep learning and transfer learning models like CNN, VGG16, ResNet50, MobileNetV2, and DenseNet.   The best-performing model (VGG16) is deployed using Streamlit for real-time predictions.

Problem Statement
Manually identifying fish species from images is tedious and error-prone.  
The goal of this project is to automate fish classification using computer vision techniques — providing a scalable and efficient solution for fisheries, researchers, and marine studies.

Objectives
- Develop a deep learning model to classify multiple fish species.
- Evaluate and compare performance of multiple architectures.
- Deploy the most accurate model using Streamlit for real-time use.


Dataset Description
- Dataset contains images of 11 fish species in separate folders (one folder per class).  
- Each image is labeled automatically through folder structure.  
- Data preprocessing includes resizing, normalization, and augmentation. 


Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| Language | Python |
| Framework | PyTorch |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Transfer Learning Models | VGG16, ResNet50, MobileNetV2, DenseNet |
| Utilities | Torchvision, PIL, tqdm, numpy, pandas |

 Workflow

1. Data Preprocessing  
   - Loaded dataset using `ImageFolder`  
   - Applied transformations (resize, normalize, augment)  

2. Model Training  
   - Custom CNN & Transfer Learning (VGG16, ResNet50, MobileNetV2, DenseNet)  
   - Loss function → CrossEntropyLoss  
   - Optimizer → Adam  
   - Scheduler → ReduceLROnPlateau  

3. Model Evaluation  
   - Compared models using Accuracy, Precision, Recall, F1-score  
   - Selected the best-performing model (VGG16)  

4. Deployment  
   - Streamlit app for live prediction  
   - Upload image → Predict species → Show confidence %

Training Metrics
- Plotted **Loss vs Epoch** and Accuracy vs Epoch graphs.  
- Used confusion matrix and classification report for detailed performance. 

