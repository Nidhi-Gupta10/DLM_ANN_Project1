import pandas as pd
from sklearn.model_selection import train_test_split
from model import build_ann, train_model  # Assuming model.py contains ANN functions
def custom_preprocess(dataframe, target_variable, train_test_split_percentage=80):
    """
    Performs data preprocessing, specifically splitting into train and test sets.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        target_variable (str): The name of the target variable column.
        train_test_split_percentage (int, optional): Percentage of data for training. Defaults to 80.

    Returns:
        tuple: X_train, X_test, y_train, y_test (pandas DataFrames/Series).
    """

    y = dataframe[target_variable]
    X = dataframe.drop(columns=[target_variable])

    train_size = train_test_split_percentage / 100.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42  # Added random_state for reproducibility
    )

    return X_train, X_test, y_train, y_test
def load_and_train_model(DATASET_FILE: str, target_variable: str, hyperparams: dict) -> tuple:
    """
    Loads the dataset, preprocesses it, and trains an ANN model using given hyperparameters.
    
    Args:
        DATASET_FILE (str): Path to the dataset file.
        hyperparams (dict): Dictionary containing ANN hyperparameters.
    
    Returns:
        model: Trained ANN model.
        history: Training history containing loss and accuracy metrics.
    """
    
    # Load Dataset from local environment
    df = pd.read_csv(DATASET_FILE)
    
    # Initialize Data Preprocessor
  



    X_train, X_test, y_train, y_test = custom_preprocess(df, "Depression",)
    
    # Build ANN Model
    model = build_ann(
        input_shape=X_train.shape[1],
        num_layers=hyperparams.get("num_layers", 2),
        neurons_per_layer=hyperparams.get("neurons_per_layer", 32),
        activation=hyperparams.get("activation", "ReLU"),
        dropout_rate=hyperparams.get("dropout_rate", 0.2),
        weight_init=hyperparams.get("weight_init", "he_normal"),
        batch_norm=hyperparams.get("batch_norm", True),
        l1_reg=hyperparams.get("l1_reg", 0.0),
        l2_reg=hyperparams.get("l2_reg", 0.0),
        dropconnect=hyperparams.get("dropconnect", False),
        activation_reg=hyperparams.get("activation_reg", 0.0),
        optimizer=hyperparams.get("optimizer", "Adam"),
        learning_rate=hyperparams.get("learning_rate", 0.001),
        momentum=hyperparams.get("momentum", 0.9),
        learning_rate_decay=hyperparams.get("learning_rate_decay", 0.0),
        gradient_clipping=hyperparams.get("gradient_clipping", 0.0),
        backprop_type=hyperparams.get("backprop_type", "Stochastic Gradient Descent")
    )
    
    # Train Model
    history = train_model(
        model,
        X_train, y_train,
        X_test, y_test,
        batch_size=hyperparams.get("batch_size", 32),
        epochs=hyperparams.get("epochs", 50),
        early_stopping=hyperparams.get("early_stopping", True)
    )
    
    return model, history, X_test, y_test
