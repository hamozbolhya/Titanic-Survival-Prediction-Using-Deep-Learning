import pandas as pd

def load_and_preprocess_data():
    """
    Load Titanic dataset from CSV, preprocess, and return train/validation/test splits.
    """
    # Load dataset from CSV file
    df = pd.read_csv("../data/dataset.csv")  # Load from /data folder

    # Drop rows where target (survived) is missing
    df = df.dropna(subset=['Survived'])

    # Fill missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    df['Sex'] = encoder.fit_transform(df['Sex'])  # Male → 1, Female → 0
    df['Embarked'] = encoder.fit_transform(df['Embarked'])  # S → 2, C → 0, Q → 1

    # Select features and target variable
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
    target = 'Survived'
    df = df[features + [target]]

    # Convert to float32 for TensorFlow
    import numpy as np
    df = df.astype(np.float32)

    # Split into train/validation/test sets
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Normalize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train.shape[1]
