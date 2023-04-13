# Make a function named get_titanic_data that returns the titanic data from the codeup data science database as a pandas data frame.
# Obtain your data from the Codeup Data Science Database.

import pandas as pd
from env import host, username, password

def get_titanic_data():
  
    # create the connection url
    url = f'mysql+pymysql://{username}:{password}@{host}/titanic_db'

    # read the SQL query into a DataFrame
    query = 'SELECT * FROM passengers'
    df_titanic = pd.read_sql(query, url)
    return df_titanic

df_titanic = get_titanic_data()

# Make a function named get_iris_data that returns the data from the iris_db on the codeup data science database as a pandas data frame.
# The returned data frame should include the actual name of the species in addition to the species_ids. 
# Obtain your data from the Codeup Data Science Database.

def get_iris_data():
  
    # create the connection url
    url = f'mysql+pymysql://{username}:{password}@{host}/iris_db'

    # read the SQL query into a DataFrame
    query = '''
            SELECT species_id, species_name, sepal_length, sepal_width, petal_length, petal_width
            FROM measurements
            JOIN species USING(species_id)
            '''
    df = pd.read_sql(query, url)
    return df

df_iris = get_iris_data()


# Make a function named get_telco_data that returns the data from the telco_churn database in SQL.
# In your SQL, be sure to join contract_types, internet_service_types, payment_types tables with the customers table,
#  so that the resulting dataframe contains all the contract, payment, and internet service options. 
# Obtain your data from the Codeup Data Science Database.

def get_telco_data():
   
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

df_telco = get_telco_data()


# Once you've got your get_titanic_data, get_iris_data, and get_telco_data functions written, now it's time to add caching to them. 
# To do this, edit the beginning of the function to check for the local filename of telco.csv, titanic.csv, or iris.csv. 
# If they exist, use the .csv file. If the file doesn't exist, then produce the SQL and pandas necessary to create a dataframe, 
# then write the dataframe to a .csv file with the appropriate name.
import os

def get_titanic_data():
    '''
    This function reads the titanic data from the Codeup data science database into a pandas DataFrame.
    If titanic.csv exists in the current directory, the function returns the data from the CSV file.
    If titanic.csv does not exist, the function produces the SQL and pandas necessary to create a DataFrame,
    writes the DataFrame to a CSV file, then returns the DataFrame.
    '''
    filename = 'titanic.csv'

    # if the file exists, read the data from the CSV file and return a DataFrame
    if os.path.isfile(filename):
        df_titanic = pd.read_csv(filename, index_col=0)
        return df_titanic
    else:
        # create the connection url
        url = f'mysql+pymysql://{username}:{password}@{host}/titanic_db'

        # read the SQL query into a DataFrame
        query = 'SELECT * FROM passengers'
        df_titanic = pd.read_sql(query, url)

        # write the DataFrame to a CSV file
        df_titanic.to_csv(filename)

        return df_titanic


def get_iris_data():
    '''
    This function reads the iris data from the Codeup data science database into a pandas DataFrame.
    If iris.csv exists in the current directory, the function returns the data from the CSV file.
    If iris.csv does not exist, the function produces the SQL and pandas necessary to create a DataFrame,
    writes the DataFrame to a CSV file, then returns the DataFrame.
    '''
    filename = 'iris.csv'

    # if the file exists, read the data from the CSV file and return a DataFrame
    if os.path.isfile(filename):
        df_iris = pd.read_csv(filename, index_col=0)
        return df_iris
    else:
        # create the connection url
        url = f'mysql+pymysql://{username}:{password}@{host}/iris_db'

        # read the SQL query into a DataFrame
        query = '''
                SELECT species_id, species_name, sepal_length, sepal_width, petal_length, petal_width
                FROM measurements
                JOIN species USING(species_id)
                '''
        df_iris = pd.read_sql(query, url)

        # write the DataFrame to a CSV file
        df_iris.to_csv(filename)

        return df_iris


def get_telco_data():
    '''
    This function reads the telco churn data from the Codeup data science database into a pandas DataFrame.
    If telco.csv exists in the current directory, the function returns the data from the CSV file.
    If telco.csv does not exist, the function produces the SQL and pandas necessary to create a DataFrame,
    writes the DataFrame to a CSV file, then returns the DataFrame.
    '''
    filename = 'telco.csv'

    # if the file exists, read the data from the CSV file and return a DataFrame
    if os.path.isfile(filename):
        df_telco = pd.read_csv(filename, index_col=0)
        return df_telco
    else:
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
        df_telco = pd.read_sql(query, url)

        # write the DataFrame to a CSV file
        df_telco.to_csv(filename)

        return df_telco

