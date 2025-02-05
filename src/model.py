import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

def create_model(input_shape):
    """
    Create and return a compiled ANN model.
    
    Parameters:
    - input_shape: int, number of input features.
    
    Returns:
    - Compiled ANN model.
    """

    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
