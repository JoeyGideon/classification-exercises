import pandas as pd

def prep_iris(df):
    '''
    This function accepts the untransformed iris data as a pandas DataFrame and applies the following transformations:
    - Drops the species_id column
    - Renames the species_name column to species
    - Creates dummy variables of the species column and concatenates them onto the original DataFrame
    '''
    # Drop the species_id column from the DataFrame
    df = df.drop(columns=['species_id'])

    # Rename the species_name column to species
    df = df.rename(columns={'species_name': 'species'})

    # Create dummy variables of the species column
    species_dummies = pd.get_dummies(df['species'], drop_first=True)

    # Concatenate the dummy variables onto the original DataFrame
    df = pd.concat([df, species_dummies], axis=1)

    return df

def prep_titanic(df):

    # Drop unnecessary columns
    df = df.drop(columns=['deck', 'embark_town', 'class'])
    
    # Fill missing values
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    
    # Encode categorical columns
    df['sex'] = pd.get_dummies(df['sex'], drop_first=True)
    df = pd.concat([df,pd.get_dummies(df['embarked'], prefix='embarked', drop_first=True)], axis=1)
    
    # Drop original categorical columns that have been encoded
    df = df.drop(columns=['embarked'])
    
    # Return the updated DataFrame
    return df


# Prepare function
def prep_telco(df):
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
            'online_security', 'online_backup', 'device_protection', 'tech_support',
            'streaming_tv', 'streaming_movies', 'paperless_billing',
            'contract_type', 'internet_service_type', 'payment_type']

    dummy_df = pd.get_dummies(df[cat_cols], drop_first=True)

    df = pd.concat([df, dummy_df], axis=1)

    df.drop(cat_cols, axis=1, inplace=True)

    return df



import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target):
    '''
    This function takes in a dataframe and target name and splits the data into
    train, validate, and test subsets. It returns the X_train, X_validate, X_test,
    y_train, y_validate, and y_test dataframes and series.
    '''
    # Split the data into train + validate and test sets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123, stratify=df[target])

    # Split the train + validate set into train and validate sets
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123, stratify=train_validate[target])

    # Separate the target variable from the features
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, X_validate, X_test, y_train, y_validate, y_test



