import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np  # Add this line at the beginning of your script


# Load data from a CSV file
def load_data(csv_file):
    return pd.read_csv(csv_file)

# Preprocess the data (you might need to adapt this based on your actual data structure and needs)
def preprocess_data(df):
    # Example: Convert categorical variables to numerical
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'Other': 2})
    df['TimeOfDay'] = df['TimeOfDay'].map({'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3})
    # Convert PreviousGlucoseLevels from string to numerical list and calculate its mean
    df['PrevGlucoseMean'] = df['PreviousGlucoseLevels'].apply(lambda x: pd.to_numeric(x.split(','))).apply(np.mean)
    # Drop the original PreviousGlucoseLevels column
    df = df.drop('PreviousGlucoseLevels', axis=1)
    return df

# TO DOs: Manu, Diego: This is a simplified example. You should refine the preprocessing and model training based on your data and goals.
# TO DOs: Make sure to handle missing values appropriately, which might involve imputation or removal of missing data.
# The threshold for labeling a glucose level as 'high' (in this case, >180 mg/dL) should be determined based on medical guidelines or consultation with medical professionals.

# Main function to train and predict
def main(csv_file='your_dataset.csv'):
    data = load_data(csv_file)
    data = preprocess_data(data)

    X = data.drop('GlucoseLevel', axis=1)
    y = data['GlucoseLevel'].apply(lambda x: 1 if x > 180 else 0)  # Example threshold for high glucose level

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')


if __name__ == '__main__':
    main()
