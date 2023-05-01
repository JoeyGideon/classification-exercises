import pandas as pd
from env import host, username, password
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
import pandas as pd
def get_telco_data():
    """
    This function connects to the telco_churn database and retrieves data from the customers, contract_types,
    internet_service_types, and payment_types tables. The resulting DataFrame contains all columns from these
    tables and is returned by the function.
    """
   
    # create the connection url
    url = f'mysql+pymysql://{username}:{password}@{host}/telco_churn'

    # read the SQL query into a DataFrame
    query = '''
            SELECT *
            FROM customers
            JOIN contract_types USING(contract_type_id)
            JOIN internet_service_types USING(internet_service_type_id)
            JOIN payment_types USING(payment_type_id)
            '''
    df = pd.read_sql(query, url)

    return df

def prep_telco(df):
    """
    This function prepares a telco DataFrame for machine learning modeling by performing the following steps:
    - Drops duplicate rows
    - Replaces blank values with NaN
    - Replaces "No internet service" with "No" for relevant columns
    - Replaces binary columns with 1 (Yes) and 0 (No)
    - Converts total_charges to a numeric data type
    - Encodes categorical columns using one-hot encoding
    - Drops original categorical columns

    Parameters:
    df (pandas DataFrame): The telco DataFrame to be prepared.

    Returns:
    pandas DataFrame: The prepared telco DataFrame.
    """
    
    # Drop duplicate rows
    df = df.drop_duplicates()

    # Replace blank values with NaN
    df = df.replace("", pd.np.nan)

    # Replace "No internet service" with "No" for relevant columns
    cols = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
    for col in cols:
        df[col] = df[col].replace("No internet service", "No")

    # Replace binary columns with 1 (Yes) and 0 (No)
    binary_cols = ['partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn']
    for col in binary_cols:
        df[col] = df[col].replace({"Yes": 1, "No": 0})

    # Convert total_charges to a numeric data type
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    
    # Encode the categorical columns
    cat_cols = ['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines',
            'online_security', 'online_backup', 'device_protection',
            'streaming_tv', 'streaming_movies', 'paperless_billing',
            'contract_type', 'internet_service_type', 'payment_type']

    dummy_df = pd.get_dummies(df[cat_cols], drop_first=True)

    df = pd.concat([df, dummy_df], axis=1)

    df.drop(cat_cols, axis=1, inplace=True)

    return df

def train_validate_test_split(prep_telco, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(prep_telco, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=prep_telco[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test


def calculate_baseline(prep_telco, target_col):
    # Determine the most prevalent class in the target column
    mode = prep_telco[target_col].mode()[0]

    # Make all predictions using the most prevalent class
    baseline_preds = np.full(prep_telco.shape[0], mode)

    # Calculate the accuracy of the baseline model
    baseline_accuracy = (baseline_preds == prep_telco[target_col]).mean()

    # Print the baseline prediction and accuracy
    print(f"Baseline Prediction: {mode}")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")





def evaluate_models(train_data, val_data, test_data, feature_cols, target_col):
    # Split the data into feature and target columns
    X_train, y_train = train_data[feature_cols], train_data[target_col]
    X_val, y_val = val_data[feature_cols], val_data[target_col]
    X_test, y_test = test_data[feature_cols], test_data[target_col]

    # Create four different models
    models = [
        ('Logistic Regression', LogisticRegression(random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Random Forest', RandomForestClassifier(random_state=42)),
        ('Support Vector Machine', SVC(random_state=42))
    ]

    # Evaluate each model on the training and validation data
    train_val_scores = []
    for name, model in models:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)

        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)

        train_val_scores.append((name, train_accuracy, train_precision, train_recall, train_f1, val_accuracy, val_precision, val_recall, val_f1))
        
    # Evaluate the best model on the testing data
    best_model_name = None
    best_model_accuracy = 0
    for name, model in models:
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if accuracy > best_model_accuracy:
            best_model_name = name
            best_model_accuracy = accuracy
            best_model_scores = (accuracy, precision, recall, f1)

    # Print the scores for each model on the train and validate sets
    print("Train and Validation Metrics:")
    for name, train_accuracy, train_precision, train_recall, train_f1, val_accuracy, val_precision, val_recall, val_f1 in train_val_scores:
        print(f"{name}:")
        print(f"\tTrain Accuracy: {train_accuracy:.4f}")
        print(f"\tTrain Precision: {train_precision:.4f}")
        print(f"\tTrain Recall: {train_recall:.4f}")
        print(f"\tTrain F1-Score: {train_f1:.4f}")
        print(f"\tValidation Accuracy: {val_accuracy:.4f}")
        print(f"\tValidation Precision: {val_precision:.4f}")
        print(f"\tValidation Recall: {val_recall:.4f}")
        print(f"\tValidation F1-Score: {val_f1:.4f}")

    # Print the scores for the best model on the test set
    print(f"Best model on Test set: {best_model_name}")
    print(f"\tAccuracy: {best_model_scores[0]:.4f}")
    print(f"\tPrecision: {best_model_scores[1]:.4f}")
    print(f"\tRecall: {best_model_scores[2]:.4f}")
    print(f"\tF1-Score: {best_model_scores[3]:.4f}")



# This code trains and evaluates each model on the training and validation sets,
#  but only evaluates the best model on the test set. 
# The best model is determined by the highest accuracy score on the test set. 
# The scores for the best model on the test set are then printed.





