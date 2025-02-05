import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from preprocess import load_and_preprocess_data
from model import create_model  # Import the ANN model

# ================================
# 1️⃣ Load and Preprocess Data
# ================================
X_train, X_valid, X_test, y_train, y_valid, y_test, input_shape = load_and_preprocess_data()

# ================================
# 2️⃣ Define and Compile the Model
# ================================
model = create_model(input_shape)

# Show model summary
model.summary()

# ================================
# 3️⃣ Train the Model with Early Stopping
# ================================
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)

# ================================
# 4️⃣ Evaluate Model Performance
# ================================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# ================================
# 5️⃣ Plot Training Progress
# ================================

# Plot accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Model Accuracy Over Epochs")
plt.show()

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Model Loss Over Epochs")
plt.show()
