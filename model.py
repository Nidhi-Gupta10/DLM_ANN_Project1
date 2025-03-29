import tensorflow as tf
from tensorflow.keras import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam, SGD, RMSprop  # type: ignore
from tensorflow.keras.regularizers import l1_l2 # type: ignore

def build_ann(input_shape, num_layers=2, neurons_per_layer=32, activation="ReLU", dropout_rate=0.2, 
              weight_init="he_normal", batch_norm=True, l1_reg=0.0, l2_reg=0.0, dropconnect=False, 
              activation_reg=0.0, optimizer="Adam", learning_rate=0.001, momentum=0.9, 
              learning_rate_decay=0.0, gradient_clipping=0.0, backprop_type="Stochastic Gradient Descent"):
    """
    Builds an Artificial Neural Network (ANN) model with given hyperparameters.

    Args:
        input_shape (int): Number of input features.
        num_layers (int): Number of hidden layers.
        neurons_per_layer (int): Number of neurons per hidden layer.
        activation (str): Activation function (ReLU, Sigmoid, Tanh, LeakyReLU).
        dropout_rate (float): Dropout rate to prevent overfitting.
        weight_init (str): Weight initialization method.
        batch_norm (bool): Apply batch normalization.
        l1_reg (float): L1 regularization strength.
        l2_reg (float): L2 regularization strength.
        dropconnect (bool): Apply DropConnect regularization.
        activation_reg (float): Activation regularization strength.
        optimizer (str): Optimizer to use (Adam, SGD, RMSprop).
        learning_rate (float): Learning rate for the optimizer.
        momentum (float): Momentum for optimizers like SGD.
        learning_rate_decay (float): Learning rate decay factor.
        gradient_clipping (float): Maximum norm for gradient clipping.
        backprop_type (str): Type of backpropagation (BGD, SGD, MBGD).
    
    Returns:
        model: Compiled ANN model.
    """
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation=activation.lower(), kernel_initializer=weight_init, 
                    kernel_regularizer=l1_l2(l1_reg, l2_reg), input_shape=(input_shape,)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    for _ in range(num_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation.lower(), kernel_initializer=weight_init, 
                        kernel_regularizer=l1_l2(l1_reg, l2_reg)))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation="sigmoid"))  # Binary classification output
    
    optimizers = {
        "Adam": Adam(learning_rate, decay=learning_rate_decay, clipnorm=gradient_clipping),
        "SGD": SGD(learning_rate, momentum=momentum, decay=learning_rate_decay, clipnorm=gradient_clipping),
        "RMSprop": RMSprop(learning_rate, decay=learning_rate_decay, clipnorm=gradient_clipping)
    }
    
    model.compile(loss="binary_crossentropy", optimizer=optimizers.get(optimizer, Adam(learning_rate)), metrics=["accuracy"])
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=50, early_stopping=True):
    """
    Trains the ANN model with given hyperparameters.
    
    Args:
        model: Compiled ANN model.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        early_stopping (bool): Apply early stopping.
    
    Returns:
        history: Training history containing loss and accuracy metrics.
    """
    callbacks = []
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True))
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks)
    
    return history
