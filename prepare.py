import pandas as pd

def prep_iris(df):
    '''
    This function accepts the untransformed iris data as a pandas DataFrame and applies the following transformations:
    - Drops the species_id and measurement_id columns
    - Renames the species_name column to species
    - Creates dummy variables of the species column and concatenates them onto the original DataFrame
    '''
    # Drop the species_id and measurement_id columns from the DataFrame
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
    df = df.drop(columns=['deck', 'embark_town', 'survived', 'class'])
    
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

import pandas as pd

def prep_telco(raw_telco_data):
    # Drop any unnecessary, unhelpful, or duplicated columns
    df_telco = raw_telco_data.drop(['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], axis=1)

    # Get the categorical columns
    cat_cols = ['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines',
                'online_security', 'online_backup', 'device_protection', 'tech_support',
                'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn',
                'contract_type', 'internet_service_type', 'payment_type']

    # Use get_dummies() to create dummy variables
    dummy_df = pd.get_dummies(df_telco[cat_cols], drop_first=True)

    # Concatenate the dummy variables onto the original DataFrame
    df_telco = pd.concat([df_telco, dummy_df], axis=1)

    # Drop the original categorical columns
    df_telco.drop(cat_cols, axis=1, inplace=True)

    # Return the cleaned and encoded DataFrame
    return df_telco



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

