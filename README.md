# 🧠 Neural Network Playground

An interactive **Neural Network Playground** built with **Python, Streamlit, TensorFlow/Keras, and Scikit-learn**.

This application allows users to experiment with different neural network configurations, datasets, training parameters, activation functions, and regularization techniques through an interactive web interface.

## 🚀 Project Overview

The application provides an interactive environment for understanding how neural networks behave under different configurations.

Users can:

* Select a dataset from GitHub
* Upload their own CSV dataset
* Choose between Classification and Regression
* Adjust the train-test ratio
* Change the learning rate
* Select activation functions
* Configure hidden layers and neurons
* Select batch size and number of epochs
* Apply L1 or L2 regularization
* Train a neural network interactively
* View model performance
* Visualize training and validation loss
* Visualize decision boundaries for classification problems

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **TensorFlow / Keras**
* **Scikit-learn**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **MLxtend**

## ⚙️ Features

### 📊 Dataset Selection

The application supports predefined datasets hosted on GitHub, including:

* U-Shape
* Concentric Circles
* Linear Separation
* Outlier
* Overlap
* XOR
* Two Spirals
* Random

Users can also upload their own CSV dataset.

### 🧠 Neural Network Configuration

Users can interactively configure:

| Parameter           | Description                                   |
| ------------------- | --------------------------------------------- |
| Problem Type        | Classification or Regression                  |
| Train-Test Ratio    | Controls training/testing split               |
| Learning Rate       | Controls optimizer step size                  |
| Activation Function | ReLU, Sigmoid, Tanh, or Linear                |
| Hidden Layers       | Number of neural network hidden layers        |
| Neurons             | Number of neurons per hidden layer            |
| Batch Size          | Number of samples processed per training step |
| Epochs              | Number of training iterations                 |
| Regularization      | None, L1, or L2                               |
| Regularization Rate | Controls regularization strength              |

## 🔄 Machine Learning Workflow

The application follows this workflow:

```text
Dataset Selection
       ↓
Data Loading
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Neural Network Creation
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Visualization
```

## 📈 Model Evaluation

### Classification

For classification problems, the application displays:

* Test Loss
* Test Accuracy
* Decision Boundary

### Regression

For regression problems, the application displays:

* Test Loss
* Mean Squared Error (MSE)

## 📊 Visualizations

The application generates:

* Training Loss
* Validation/Test Loss
* Classification Decision Boundaries

These visualizations help users understand model training and classification behavior.

## 🧪 Regularization

The application supports two regularization techniques:

### L1 Regularization

Helps encourage sparse model weights.

### L2 Regularization

Helps reduce excessively large model weights and can help control overfitting.

## 📁 Project Structure

```text
nueral-network-project/
│
├── datasab/
│   ├── 1.ushape.csv
│   ├── 2.concentriccir1.csv
│   ├── 3.concentriccir2.csv
│   ├── 4.linearsep.csv
│   ├── 5.outlier.csv
│   ├── 6.overlap.csv
│   ├── 7.xor.csv
│   ├── 8.twospirals.csv
│   └── 9.random.csv
│
├── app.py
├── README.md
└── requirements.txt
```

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/moinuddinkhaja/nueral-network-project.git
```

### 2. Navigate to the project directory

```bash
cd nueral-network-project
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 💡 Learning Outcomes

This project demonstrates practical understanding of:

* Artificial Neural Networks
* TensorFlow/Keras
* Classification and Regression
* Train-Test Splitting
* Feature Scaling
* Activation Functions
* Model Training
* Hyperparameter Configuration
* L1/L2 Regularization
* Model Evaluation
* Decision Boundary Visualization
* Streamlit Application Development

## 🔮 Future Improvements

Potential improvements include:

* Add additional datasets
* Implement noise-level functionality
* Add confusion matrix and classification metrics
* Add regression metrics such as MAE and R²
* Add model architecture visualization
* Add real-time prediction functionality
* Deploy the Streamlit application publicly

## 👨‍💻 Author

**Mohammed Khaja Moinuddin**

GitHub: `moinuddinkhaja`

## 📌 Project Type

**Machine Learning | Deep Learning | Neural Networks | Streamlit | Interactive ML Application**
