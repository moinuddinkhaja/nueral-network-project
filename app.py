import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

st.set_page_config(page_title="A Neural Network Playground")
st.title("Neural Network Playground")

# Fix for dataset URL concatenation
def get_dataset(name_or_file):
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

    if name_or_file in file_paths:
        try:
            url = base_url + file_paths[name_or_file]
            data = pd.read_csv(url, header=None)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            return X, y
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None, None
    elif hasattr(name_or_file, 'read'):
        try:
            data = pd.read_csv(name_or_file, header=None)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            return X, y
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return None, None
    else:
        st.error(f"Dataset name not recognized: {name_or_file}")
        return None, None

# Model creation function
def create_model(input_dim, learning_rate, activation, num_layers, num_neurons, regularization, reg_rate, problem_type):
    model = Sequential()
    for _ in range(num_layers):
        if regularization == "L1":
            model.add(Dense(num_neurons, input_dim=input_dim, activation=activation,
                            kernel_regularizer=tf.keras.regularizers.l1(reg_rate)))
        elif regularization == "L2":
            model.add(Dense(num_neurons, input_dim=input_dim, activation=activation,
                            kernel_regularizer=tf.keras.regularizers.l2(reg_rate)))
        elif regularization == "L1 & L2":
            model.add(Dense(num_neurons, input_dim=input_dim, activation=activation,
                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_rate, l2=reg_rate)))
        else:
            model.add(Dense(num_neurons, input_dim=input_dim, activation=activation))
        input_dim = num_neurons

    if problem_type == "Classification":
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="mean_squared_error", metrics=["mse"])

    return model

# Sidebar settings
st.sidebar.title("Neural Network Configuration")
dataset_name = st.sidebar.selectbox("Select Dataset", ["ushape", "concentriccir1", "concentriccir2", "linearsep", "outlier", "overlap", "xor", "twospirals", "random"])
problem_type = st.sidebar.selectbox("Problem Type", ["Classification", "Regression"])
train_test_ratio = st.sidebar.slider("Ratio of Training to Test Data", 0.1, 0.9, 0.8)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
activation = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "linear"])
num_layers = st.sidebar.slider("Number of Hidden Layers", 1, 50, 5)
num_neurons = st.sidebar.slider("Number of Neurons per Layer", 1, 100, 2)
batch_size = st.sidebar.slider("Batch Size", 1, 100, 10)
epochs = st.sidebar.slider("Epochs", 1, 1000, 100)
regularization = st.sidebar.selectbox("Regularization", ["None", "L1", "L2", "L1 & L2"])
reg_rate = st.sidebar.slider("Regularization Rate", 0.0, 1.0, 0.0)

# Submit button to start training
if st.sidebar.button("Submit"):
    X, y = get_dataset(dataset_name)
    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_test_ratio, random_state=42)

        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = create_model(X_train.shape[1], learning_rate, activation, num_layers, num_neurons, regularization, reg_rate, problem_type)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))

        if problem_type == "Classification":
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            st.write(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        else:
            loss, mse = model.evaluate(X_test, y_test, verbose=0)
            st.write(f"Test Loss: {loss:.4f}, Test MSE: {mse:.4f}")

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
