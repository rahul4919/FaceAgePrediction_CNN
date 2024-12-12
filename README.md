
# Age Prediction Using EfficientNet-B4

This project implements an age prediction model using a pretrained `EfficientNet-B4` network. The model is trained on a dataset of facial images and predicts the approximate age of a person based on their photo.

---

## Features
- Custom PyTorch Dataset class for image preprocessing and annotation handling.
- Use of EfficientNet-B4 with a modified classification head for regression.
- Optimized training with Adam optimizer and MAE loss function.
- Supports training and testing datasets with easy extensibility.

---

## Project Structure
```
.
├── train.csv              # Training dataset annotations
├── test.csv               # Testing dataset annotations
├── train/                 # Folder containing training images
├── test/                  # Folder containing test images
├── model.py               # Contains the EfficientNet-based model
├── dataset.py             # Custom PyTorch Dataset class
├── train.py               # Training script
├── predict.py             # Prediction script
├── submission.csv         # Submission file generated after predictions
└── README.md              # Project documentation
```

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- PyTorch
- torchvision
- pandas
- numpy
- tqdm
- matplotlib
- scikit-learn
- PIL (Pillow)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/age-prediction.git
   cd age-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Training the Model
1. Organize the training and testing datasets:
   ```
   - train/
   - test/
   - train.csv
   - test.csv
   ```
2. Run the training script:
   ```bash
   python train.py
   ```

---

## Predicting Ages
1. Use the trained model to make predictions on the test dataset:
   ```bash
   python predict.py
   ```

2. The predictions will be saved in `submission.csv`.

---

## Customization
- Modify hyperparameters in the `train.py` script:
  - `batch_size`
  - `learning_rate`
  - `num_epochs`
- Extend the dataset class (`AgeDataset`) to include additional data augmentation or preprocessing.

---

## Acknowledgements
- [PyTorch](https://pytorch.org/) for deep learning utilities.
- [EfficientNet](https://arxiv.org/abs/1905.11946) for the pretrained architecture.

---

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
