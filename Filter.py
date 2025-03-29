import streamlit as st

def hyperparameter_filters():
    """
    Function to create a sidebar filter panel for hyperparameter tuning in Streamlit.
    Returns a dictionary with selected hyperparameters.
    """
    st.sidebar.header("🔧 Hyperparameter Tuning")
    
    # Model Architecture
    num_layers = st.sidebar.slider("Number of Hidden Layers", 1, 10, 2)
    neurons_per_layer = st.sidebar.slider("Neurons per Layer", 8, 256, 32, step=8)
    activation = st.sidebar.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"], index=0)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.7, 0.2, step=0.05)
    weight_init = st.sidebar.selectbox("Weight Initialization", ["he_normal", "glorot_uniform", "random_normal"], index=0)
    batch_norm = st.sidebar.checkbox("Apply Batch Normalization", value=True)
    
    # Regularization
    l1_reg = st.sidebar.slider("L1 Regularization", 0.0, 0.1, 0.0, step=0.01)
    l2_reg = st.sidebar.slider("L2 Regularization", 0.0, 0.1, 0.0, step=0.01)
    dropconnect = st.sidebar.checkbox("Apply DropConnect", value=False)
    activation_reg = st.sidebar.slider("Activation Regularization", 0.0, 0.1, 0.0, step=0.01)
    
    # Training Parameters
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128, 256], index=1)
    learning_rate = st.sidebar.slider("Learning Rate", 0.00001, 0.1, 0.001, format="%.5f")
    optimizer = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"], index=0)
    epochs = st.sidebar.slider("Epochs", 10, 300, 50, step=10)
    momentum = st.sidebar.slider("Momentum (for SGD)", 0.0, 0.99, 0.9, step=0.01)
    learning_rate_decay = st.sidebar.slider("Learning Rate Decay", 0.0, 0.1, 0.0, step=0.01)
    gradient_clipping = st.sidebar.slider("Gradient Clipping", 0.0, 5.0, 0.0, step=0.1)
    early_stopping = st.sidebar.checkbox("Enable Early Stopping", value=True)
    
    # Backpropagation Type
    backprop_type = st.sidebar.selectbox("Backpropagation Type", ["Batch Gradient Descent", "Stochastic Gradient Descent", "Mini-Batch Gradient Descent"], index=1)
    
    # Collect all hyperparameters
    hyperparams = {
        "num_layers": num_layers,
        "neurons_per_layer": neurons_per_layer,
        "activation": activation,
        "dropout_rate": dropout_rate,
        "weight_init": weight_init,
        "batch_norm": batch_norm,
        "l1_reg": l1_reg,
        "l2_reg": l2_reg,
        "dropconnect": dropconnect,
        "activation_reg": activation_reg,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "epochs": epochs,
        "momentum": momentum,
        "learning_rate_decay": learning_rate_decay,
        "gradient_clipping": gradient_clipping,
        "early_stopping": early_stopping,
        "backprop_type": backprop_type
    }
    
    return hyperparams