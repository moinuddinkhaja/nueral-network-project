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

# Set page configuration with a descriptive title and icon
st.set_page_config(page_title="Interactive Neural Network Playground: Visualize and Tinker with Models in Real-Time", page_icon="🍭")

# Add modern theme background with clean design
page_bg_img = '''
<style>
body {
    background: #f0f2f6;
    color: #333;
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
    color: #333;
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

code {
    color: #d63384;
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
    # Base URL for raw content in GitHub repository
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

def create_model(input_dim, learning_rate, activation, num_layers, num_neurons, regularization, reg_rate, problem_type):
    model = Sequential()
    
    if regularization == "L1":
        model.add(Dense(num_neurons, input_dim=input_dim, activation=activation, 
                        kernel_regularizer=regularizers.l1(reg_rate)))
    elif regularization == "L2":
        model.add(Dense(num_neurons, input_dim=input_dim, activation=activation, 
                        kernel_regularizer=regularizers.l2(reg_rate)))
    else:
        model.add(Dense(num_neurons, input_dim=input_dim, activation=activation))
    
    for _ in range(num_layers - 1):
        if regularization == "L1":
            model.add(Dense(num_neurons, activation=activation, 
                            kernel_regularizer=regularizers.l1(reg_rate)))
        elif regularization == "L2":
            model.add(Dense(num_neurons, activation=activation, 
                            kernel_regularizer=regularizers.l2(reg_rate)))
        else:
            model.add(Dense(num_neurons, activation=activation))
    
    if problem_type == "Classification":
        model.add(Dense(1, activation="sigmoid"))  # Output layer for binary classification
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.add(Dense(1))  # Output layer for regression
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="mean_squared_error", metrics=["mse"])
    
    return model

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o', cmap='coolwarm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.colorbar()
    plt.show()

# Page selector
page = st.sidebar.selectbox("Select Page", ["Home", "Neural Network Playground"])

if page == "Home":
    st.title("Welcome to the Interactive Neural Network Playground")
    st.header("Explore Neural Networks Like Never Before!")
    st.write("""
    **Purpose:**  
    The Interactive Neural Network Playground is a web application designed to help you visualize, configure, and experiment with neural network models in real-time. Whether you're a data science enthusiast, a student, or a researcher, this playground offers an intuitive interface to dive into the world of machine learning.
    """)

    st.subheader("Features")
    st.write("""
    **1. Intuitive Interface**
    - **User-Friendly Sidebar:** Easily select datasets, configure model parameters, and manage training settings using a sleek sidebar interface.
    - **Real-Time Visualizations:** Watch your model in action with interactive plots and graphs that update as you adjust settings.

    **2. Dataset Exploration**
    - **Diverse Datasets:** Choose from a variety of pre-loaded datasets, including classic patterns like XOR, concentric circles, and more. Each dataset is designed to test different aspects of neural network performance.
    - **Custom Data Loading:** Load your own datasets and experiment with your unique data challenges.

    **3. Model Configuration**
    - **Flexible Hyperparameters:** Adjust critical model parameters such as learning rate, activation functions, number of layers, and neurons per layer.
    - **Regularization Options:** Apply L1 or L2 regularization to control overfitting and enhance model generalization.

    **4. Training and Evaluation**
    - **Train Models Easily:** Initiate model training with a single click. The playground handles data preparation, model creation, and training.
    - **Comprehensive Evaluation:** Get detailed insights into your model’s performance with metrics like accuracy, loss, and mean squared error.

    **5. Interactive Visualizations**
    - **Decision Boundary Visualization:** See how your model classifies data and learns decision boundaries with Plotly’s interactive contour plots.
    - **Training Loss Curves:** Analyze the training and validation loss over epochs to understand model learning dynamics.
    """)

    st.subheader("How It Works")
    st.write("""
    **1. Select a Dataset:** Use the sidebar to choose from various pre-loaded datasets or upload your own.

    **2. Configure Your Model:** Customize the neural network by adjusting hyperparameters such as learning rate, number of layers, and activation functions.

    **3. Train and Evaluate:** Click “Train Model” to start training your model. Once training is complete, review performance metrics and visualizations.

    **4. Explore Results:** Analyze decision boundaries and training curves to gain insights into your model’s performance.
    """)

    st.subheader("Why Use the Playground?")
    st.write("""
    **- Hands-On Learning:** Gain practical experience with neural networks and machine learning concepts.
    **- Visual Feedback:** Understand model behavior through real-time visualizations and performance metrics.
    **- Experiment Freely:** Test different configurations and datasets to see how changes impact model outcomes.
    """)

    st.subheader("Contact Us")
    st.write("""
    Have questions or need assistance? Feel free to reach out to us at:

    **Email:** [Mohameedkhajamoinuddin@gmail.com](Mohameedkhajamoinuddin@gmail.com)  
    **Support Page:** [My Blog](https://medium.com/@moinuddinkhaja70)  

    Alternatively, you can also reach out via:

    **LinkedIn:** [Chandrakanth Kunta](https://www.linkedin.com/in/khaja-moinuddin-56776a2b3/)  
    **GitHub:** [Chandrakanth Kunta](https://github.com/moinuddinkhaja)  
    """)

elif page == "Neural Network Playground":
    st.title("Neural Network Playground")
    st.header("Configure Your Neural Network")

    dataset_option = st.selectbox("Select Dataset", ["Upload your own data", "ushape", "concentriccir1", "concentriccir2", "linearsep", "outlier", "overlap", "xor", "twospirals", "random"])
    
    X, y = None, None
    
    if dataset_option == "Upload your own data":
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file:
            X, y = get_dataset(uploaded_file)
    else:
        X, y = get_dataset(dataset_option.lower())

    if X is not None and y is not None:
        st.write("Dataset loaded successfully!")
        st.write(f"Feature shape: {X.shape}")
        st.write(f"Target shape: {y.shape}")

        # Convert y to integer if the problem type is classification
        problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])
        
        if problem_type == "Classification":
            y = y.astype(np.int_)
        
        # Train-test split
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Convert y_train and y_test to integers if the problem type is classification
        if problem_type == "Classification":
            y_train = y_train.astype(np.int_)
            y_test = y_test.astype(np.int_)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Hyperparameters
        learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001)
        activation = st.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])
        num_layers = st.slider("Number of Layers", min_value=1, max_value=5, value=3)
        num_neurons = st.slider("Number of Neurons per Layer", min_value=5, max_value=100, value=50)
        regularization = st.selectbox("Regularization", ["None", "L1", "L2"])
        reg_rate = st.slider("Regularization Rate", min_value=0.0001, max_value=0.1, value=0.01)
        
        # Create and train the model
        model = create_model(X_train.shape[1], learning_rate, activation, num_layers, num_neurons, regularization, reg_rate, problem_type)

        st.subheader("Model Summary")
        model.summary(print_fn=lambda x: st.text(x))

        # Train the model
        epochs = st.slider("Epochs", min_value=1, max_value=500, value=100)
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)

        # Display training history
        st.subheader("Training History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, epochs + 1)), y=history.history['loss'], mode='lines', name='Training Loss'))
        fig.add_trace(go.Scatter(x=list(range(1, epochs + 1)), y=history.history['val_loss'], mode='lines', name='Validation Loss'))
        if problem_type == "Classification":
            fig.add_trace(go.Scatter(x=list(range(1, epochs + 1)), y=history.history['accuracy'], mode='lines', name='Training Accuracy'))
            fig.add_trace(go.Scatter(x=list(range(1, epochs + 1)), y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy'))
        fig.update_layout(title='Training and Validation Loss/Accuracy', xaxis_title='Epochs', yaxis_title='Loss/Accuracy', template='plotly_white')
        st.plotly_chart(fig)

        # Decision boundary plot (only for 2D data)
        if X_train.shape[1] == 2:
            st.subheader("Decision Boundary")
            plot_decision_boundary(model, X_train, y_train)
            st.pyplot(plt)
    else:
        st.error("Failed to load dataset.")
