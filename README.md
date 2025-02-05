# 🚢 Titanic Survival Prediction with Deep Learning

## 📌 Project Overview
This project builds an **Artificial Neural Network (ANN)** to predict whether a passenger survived the Titanic disaster using deep learning techniques.

### 🧠 Technologies Used
- **Python** (Deep Learning & Data Science)
- **TensorFlow/Keras** (For ANN Model)
- **Pandas, NumPy** (For Data Manipulation)
- **Matplotlib, Seaborn** (For Visualization)
- **Scikit-Learn** (For Data Preprocessing)
- **Jupyter Notebook** (For Experimentation)

---

## 🔹 Dataset
The dataset comes from the Titanic dataset (`titanic.csv`).  
It contains:
- **Features**: `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`
- **Target**: `Survived` (0 = No, 1 = Yes)

---

## 🚀 Installation Guide
1. **Clone the repository** (if using Git):
   ```bash
   git clone https://github.com/your-repo-name.git
   cd DeepLearning_Project


Install dependencies:

pip install -r requirements.txt
Run Jupyter Notebook:

jupyter notebook
Open notebooks/deep_learning.ipynb and follow the steps.

📂 Project Structure

DeepLearning_Project/
│── data/                   # Dataset Folder
│   ├── titanic.csv         # Titanic dataset (used for training)
│── src/                    # Source Code
│   ├── preprocess.py       # Data preprocessing script
│   ├── model.py            # ANN model definition
│   ├── train_model.py      # Training script
│── notebooks/              # Jupyter Notebooks
│   ├── deep_learning.ipynb # Full project workflow
│── reports/                # Documentation
│   ├── report.pdf          # Project report
│   ├── slides.pptx         # Presentation slides
│── requirements.txt        # Dependencies list
│── README.md               # Project documentation


📊 Training Results
✅ The ANN model was trained using 50 epochs.
✅ Achieved a test accuracy of X.XX%.
✅ Below are training and validation accuracy curves:

🔹 Accuracy Curve

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

📝 How to Train the Model
Run the training script:

python src/train_model.py
To load a trained model for predictions:

from tensorflow.keras.models import load_model
model = load_model("titanic_model.h5")





