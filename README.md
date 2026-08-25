# Neural Networks and Deep Learning Fundamentals

## Project Description

This repository contains the implementation of foundational and advanced concepts in Neural Networks and Deep Learning. The project covers theoretical derivations and practical implementations using PyTorch. Key components include:

- **Theoretical Concepts:** Implementation of the XOR function using the MADALINE architecture and weight optimization via the discrete Delta Rule.
- **Autoencoders & Classification (MNIST):** Designing Autoencoders to compress the MNIST dataset into lower-dimensional latent spaces (8 and 4 neurons). A feedforward neural network classifier is then trained on the extracted features while keeping the encoder weights frozen.
- **Regression Analysis (WHO Life Expectancy):** Predicting life expectancy using various Multilayer Perceptron (MLP) architectures. The models range from a simple 1-hidden-layer network to an advanced model utilizing Batch Normalization, Dropout, and LeakyReLU.

## Technologies Used

- **Python**
- **PyTorch**
- **Scikit-learn**
- **Pandas & NumPy**
- **Matplotlib**

## Project Structure

```text
├── Data/
│   ├── train-images.rar
│   ├── train-labels.rar
│   ├── t10k-images.rar
│   ├── t10k-labels.rar
│   └── Life Expectancy Data.csv
├── src/                  # Python scripts and Jupyter notebooks for model training
├── document.pdf          # Original assignment description and questions
├── report.pdf            # Comprehensive report detailing methodology and analysis
└── README.md             # Project documentation
```

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abolfazlghandi/NNDL-PyTorch-Fundamentals.git
   cd NNDL-PyTorch-Fundamentals
   ```
2. **Install dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install torch torchvision pandas scikit-learn matplotlib numpy
   ```
3. **Prepare the Data:**
   Ensure the MNIST dataset and the WHO Life Expectancy dataset are placed inside the `Data/` directory.
4. **Execute the code:**
   Run the provided Jupyter notebooks or Python scripts to train the Autoencoders, Classifiers, and Regression models.

## Key Results

### 1. MNIST Autoencoder & Classifier

- **Latent Dimension = 8:**
  - Autoencoder Reconstruction MSE: **0.0306**
  - Classifier Test Accuracy: **~75.68%**
- **Latent Dimension = 4:**
  - Autoencoder Reconstruction MSE: **0.0429**
  - Classifier Test Accuracy: **~73.85%**

_Conclusion:_ The 8-neuron latent space retained significantly more structural information from the images, enabling the classifier to achieve higher accuracy.

### 2. WHO Life Expectancy Regression

Three different MLP architectures were evaluated on the test set:

- **Model 1 (1 Hidden Layer, 64 neurons):** MSE = **3.55**, RMSE = 1.88, R² = **0.9570** _(Best Performance)_
- **Model 2 (2 Hidden Layers):** MSE = **4.02**, RMSE = 2.01, R² = 0.9513
- **Advanced Model (3 Hidden Layers, BatchNorm, Dropout):** MSE = **3.57**, RMSE = 1.89, R² = 0.9567

_Conclusion:_ For this relatively small tabular dataset, the simpler architecture (Model 1) provided the best Bias-Variance tradeoff, generalizing better without overfitting compared to deeper or more complex models.
