# ForgeryDetection_Hybrid_model
# Hybrid Image Forgery Detection Model

## Overview
This repository contains an advanced hybrid model for image forgery detection, integrating **Error Level Analysis (ELA)**, **Local Binary Patterns (LBP)**, **MobileNetV2**, and **CNN layers** to enhance feature extraction. Built using **TensorFlow** and **Keras**, it includes preprocessing, augmentation, and learning rate decay strategies for improved performance.

## Features
- **Custom TensorFlow Layers** for ELA and LBP
- **Hybrid Deep Learning Architecture** combining CNN and MobileNetV2
- **Image Preprocessing and Augmentation**
- **Learning Rate Decay and Early Stopping**
- **Binary Classification**: Authentic vs. Forged

## Requirements
Install dependencies using:
```bash
pip install tensorflow opencv-python numpy
```

## Model Architecture
1. **Preprocessing Layers**: Normalization, Augmentation (Flipping, Rotation, etc.)
2. **Branch 1 - MobileNetV2**: Extracts high-level features.
3. **Branch 2 - ELA & LBP Processing**:
   - **ELA**: Highlights compression artifacts.
   - **LBP**: Captures texture variations.
   - **CNN Layers**: Further process ELA outputs.
4. **Feature Fusion**: Combines both branches for final prediction.

## Training the Model
### Data Preparation
Organize your dataset as:
```
data/
    ├── class_0/  # Authentic images
    ├── class_1/  # Forged images
```
### Training
Run:
```python
python train.py
```
This script:
- Loads the dataset with augmentation.
- Trains the hybrid model.
- Saves `forgery_detection_hybrid.h5`.

## Testing the Model
Place test images in `test/` and run:
```python
test.py
```
Outputs predictions (Authentic/Forged) with confidence scores.

## Results
- High accuracy on benchmark datasets.
- Effective forgery detection through ELA and LBP features.

## Future Improvements
- More advanced backbone models.
- Explainability techniques (e.g., Grad-CAM visualization).
- Multi-class classification support.

## License
Licensed under **MIT License**.

---
### 🚀 Contribute
Submit issues, feature requests, or pull requests!

### 📫 Contact
For inquiries, reach out via GitHub Issues or email.

