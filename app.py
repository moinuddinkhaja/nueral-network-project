import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# Set page configuration
st.set_page_config(
    page_title="Interactive Neural Network Playground", 
    page_icon="🧠", 
    layout="wide"
)

# Modern theme background and CSS customization
page_bg_img = '''
<style>
body {
    background: #f8f9fa;
    color: #212529;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stApp {
    background-color: #ffffff;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    padding: 20px;
}
.sidebar .sidebar-content {
    background-color: #ffffff;
    color: #212529;
    border-right: 1px solid #ddd;
}
h1, h2, h3, h4, h5, h6 {
    color: #333;
}
button {
    background-color: #007bff;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 10px 20px;
    font-size: 16px;
}
button:hover {
    background-color: #0056b3;
}
.stCheckbox > label, .stSelectbox > div, .stSlider > div {
    color: #333;
}
.stButton > button {
    margin-top: 10px;
}
.stSelectbox, .stSlider {
    margin-top: 10px;
}
.stMarkdown {
    margin-top: 20px;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

def get_dataset(name_or_file):
    # Load dataset from URL or file
    base_url = "https://github.com/chandrakanthkunta/ANN-playground/raw/main/datasets/"
    
    file_paths = {
        "ushape": "1.ushape.csv",
        "concentriccir1": "2.concentriccir1.csv",
        "concentriccir2": "3.concentriccir2.csv",
        "linearsep": "4.linearsep.csv",
        "outlier": "5.outlier.csv",
        "overlap": "6.overlap.csv",
        "xor": "7.xor.csv",
        "twospirals": "8.twospirals.csv",
        "random": "9.random.csv"
    }

    if name_or_file in file_paths:
        url = base_url + file_paths[name_or_file]
        data = pd.read_csv(url, header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y
    else:
        data = pd.read_csv(name_or_file, header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y

def create_model(input_dim, learning_rate, activation, num_layers, num_neurons, regularization, reg_rate, problem_type):
    model = Sequential()
    
    # Add layers based on hyperparameters
    regularizer = None
    if regularization == "L1":
        regularizer = regularizers.l1(reg_rate)
    elif regularization == "L2":
        regularizer = regularizers.l2(reg_rate)
    
    model.add(Dense(num_neurons, input_dim=input_dim, activation=activation, kernel_regularizer=regularizer))
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation=activation, kernel_regularizer=regularizer))
    
    if problem_type == "Classification":
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="mean_squared_error", metrics=["mse"])
    
    return model

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o', cmap='coolwarm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.colorbar()
    st.pyplot(plt)

# Sidebar
with st.sidebar:
    st.header("Neural Network Settings")
    with st.expander("Model Parameters"):
        dataset_option = st.selectbox("Select Dataset", ["ushape", "concentriccir1", "concentriccir2", "linearsep", "outlier", "overlap", "xor", "twospirals", "random"])
        learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001)
        activation = st.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])
        num_layers = st.slider("Number of Layers", min_value=1, max_value=5, value=3)
        num_neurons = st.slider("Number of Neurons per Layer", min_value=5, max_value=100, value=50)
        regularization = st.selectbox("Regularization", ["None", "L1", "L2"])
        reg_rate = st.slider("Regularization Rate", min_value=0.0001, max_value=0.1, value=0.01)

    with st.expander("Dataset Options"):
        problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2)

st.title("Interactive Neural Network Playground")

# Load dataset
X, y = get_dataset(dataset_option)
st.write("Dataset loaded successfully!")
st.write(f"Feature shape: {X.shape}, Target shape: {y.shape}")
st.write("Here's a preview of the dataset:")
st.write(pd.DataFrame(X).head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
st.subheader("Model Training")
model = create_model(X_train.shape[1], learning_rate, activation, num_layers, num_neurons, regularization, reg_rate, problem_type)
epochs = st.slider("Epochs", min_value=1, max_value=500, value=100)

# Training with a progress bar
with st.spinner("Training the model..."):
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)

st.success("Training Complete!")

# Plot training history
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, epochs + 1)), y=history.history['loss'], mode='lines', name='Training Loss'))
fig.add_trace(go.Scatter(x=list(range(1, epochs + 1)), y=history.history['val_loss'], mode='lines', name='Validation Loss'))
if problem_type == "Classification":
    fig.add_trace(go.Scatter(x=list(range(1, epochs + 1)), y=history.history['accuracy'], mode='lines', name='Training Accuracy'))
    fig.add_trace(go.Scatter(x=list(range(1, epochs + 1)), y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy'))
fig.update_layout(title='Training and Validation Loss/Accuracy', xaxis_title='Epochs', yaxis_title='Loss/Accuracy', template='plotly_white')
st.plotly_chart(fig)

# Plot decision boundary if data is 2D
if X_train.shape[1] == 2:
    st.subheader("Decision Boundary")
    plot_decision_boundary(model, X_train, y_train)
