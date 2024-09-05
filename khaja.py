import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Set page configuration to full width
st.set_page_config(page_title="A Neural Network Playground", page_icon="🍭")

# Set the title of the web page
st.title(":rainbow[Tinker With a Neural Network Right Here in Your Browser. Don’t Worry, You Can’t Break It. We Promise.]")

# Function to fetch datasets from GitHub repository or uploaded by user
def get_dataset(name_or_file):
    # Base URL for raw content in GitHub repository
    base_url = "https://raw.githubusercontent.com/moinuddinkhaja/nueral-network-project/main/datasab/"
    
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
    
    if isinstance(name_or_file, str):
        if name_or_file in file_paths:
            url = base_url + file_paths[name_or_file]
            data = pd.read_csv(url, header=None)
        else:
            return None, None
    else:
        data = pd.read_csv(name_or_file, header=None)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# Function to create and compile a neural network model
def create_model(input_dim, learning_rate, activation, num_layers, num_neurons, regularization, reg_rate, problem_type):
    model = Sequential()
    for _ in range(num_layers):
        if regularization == "L1":
            model.add(Dense(num_neurons, input_dim=input_dim, activation=activation, 
                            kernel_regularizer=tf.keras.regularizers.l1(reg_rate)))
        elif regularization == "L2":
            model.add(Dense(num_neurons, input_dim=input_dim, activation=activation, 
                            kernel_regularizer=tf.keras.regularizers.l2(reg_rate)))
        else:
            model.add(Dense(num_neurons, input_dim=input_dim, activation=activation))
        input_dim = num_neurons
    
    if problem_type == "Classification":
        model.add(Dense(1, activation="sigmoid"))  # Output layer for binary classification
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.add(Dense(1))  # Output layer for regression
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="mean_squared_error", metrics=["mse"])
    
    return model

# Sidebar for dataset selection and model configuration
st.sidebar.title("Neural Network Configuration")

# Dataset selection: select from GitHub or upload a file
upload_option = st.sidebar.radio("Choose dataset source", ["Select from GitHub", "Upload a file"])

if upload_option == "Select from GitHub":
    dataset_name = st.sidebar.selectbox("Select Dataset", ["ushape", "concentriccir1", "concentriccir2", "linearsep", "outlier", "overlap", "xor", "twospirals", "random"])
    uploaded_file = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
    dataset_name = None

problem_type = st.sidebar.selectbox("Problem Type", ["Classification", "Regression"])
train_test_ratio = st.sidebar.slider("Ratio of Training to Test Data", 0.1, 0.9, 0.8)
noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.0)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
activation = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "linear"])
num_layers = st.sidebar.slider("Number of Hidden Layers", 1, 10, 5)
num_neurons = st.sidebar.slider("Number of Neurons per Layer", 1, 10, 2)
batch_size = st.sidebar.slider("Batch Size", 1, 100, 10)
epochs = st.sidebar.slider("Epochs", 1, 1000, 100)
regularization = st.sidebar.selectbox("Regularization", ["None", "L1", "L2"])
reg_rate = st.sidebar.slider("Regularization Rate", 0.0, 1.0, 0.0)

# Button to trigger the model building and training process
if st.sidebar.button("Submit"):
    # Load dataset based on the user's choice
    if dataset_name:
        X, y = get_dataset(dataset_name)
    elif uploaded_file:
        X, y = get_dataset(uploaded_file)
    else:
        st.write("Please select or upload a valid dataset.")
        X, y = None, None
    
    if X is not None and y is not None:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_test_ratio, random_state=42)
        
        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the model
        model = create_model(X_train.shape[1], learning_rate, activation, num_layers, num_neurons, regularization, reg_rate, problem_type)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))

        # Evaluate the model
        if problem_type == "Classification":
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            st.write(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        else:
            loss, mse = model.evaluate(X_test, y_test, verbose=0)
            st.write(f"Test Loss: {loss:.4f}, Test MSE: {mse:.4f}")

        # Plot decision regions
        plt.figure(figsize=(10, 6))
        plot_decision_regions(X_test, y_test.astype(np.integer), clf=model)
        plt.title(f"Decision Boundary for Selected Dataset")
        st.pyplot(plt)

        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        st.pyplot(plt)

    else:
        st.write("Please select a valid dataset.")
s
