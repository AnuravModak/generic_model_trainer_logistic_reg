import numpy
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import warnings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from joblib import dump

import json
import os

from joblib import load


warnings.filterwarnings('ignore')

X_train = pd.DataFrame()

y_train = pd.DataFrame()

def load_data(file_path):
    df1 = pd.read_csv(file_path)
    #print(df1)
    return df1


def data_cleaning(df1):
    df1.isnull().sum()
    df2 = df1.replace("unknown", pd.NA)
    df2.dropna(inplace=True)
    df2.reset_index(drop=True, inplace=True)
    print(df2.columns)
    return df2


def get_unique(df2):
    return df2.unique().tolist()


def check_unique(df):  # pass df['column_name']
    dup = df.duplicated(keep=False)
    df['is_duplicate'] = dup
    has_duplicates = df['is_duplicate'].any()

    if has_duplicates:
        print("There are duplicates in the DataFrame.")
        return False
    else:
        print("All values are unique in the DataFrame.")
        return True


def encode_column(df):
    try:
        arr = get_unique(df)
        output = list()
        for x in df:
            if x in arr:
                output.append(arr.index(x))
        out = pd.DataFrame(output)
        return out
    except Exception as e:
        print(f"An error occurred during encoding: {e}")
        encoder = OneHotEncoder()
        encoded_data = encoder.fit_transform(df)
        return encoded_data
def check_if_string(df):
        # Check the dtype of the column
        column_dtype = df.dtype

        # Determine if the column contains string or numeric values
        if column_dtype == 'object':
            # print(f"Column contains string values.")
            return True
        elif pd.api.types.is_numeric_dtype(df):
            # print(f"Column contains numeric values.")
            return False
        else:
            # print(f"Column contains other types of data.")
            return False


###--------------------------------------
def prerequisites(trainset_path, model_name):
    df1 = load_data(trainset_path)
    make_training_directory(model_name)
    base_path = model_name+os.path.sep

    print(df1)

    df2 = data_cleaning(df1)

    dump(df2, base_path+model_name+'_original_cleaned_df.joblib')

    df3 = encode_columns(df2)

    print(df3.head())
    return df3


def encode_columns(df2):
    columns_to_encode = df2.columns.tolist()
    df3 = df2.copy()
    for col in columns_to_encode:
        if check_if_string(df2[col]) and not check_column_if_numeric(df2[col].values):
            df3[col] = encode_column(df2[col])
    return df3


def run_logistic_regression(df3, X_train, X_test, y_train, y_test, result_column, testset_path, model_name):
    pipelines = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }
    accuracy1 = 0.0
    for name, pipeline in pipelines.items():
        print(f"Training and evaluating {name}...")
        # Train the model using the pipeline
        pipeline.fit(X_train, y_train)
        # Make predictions on the testing set
        y_pred = pipeline.predict(X_test)
        # Evaluate the classifier's performance
        accuracy1 = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy1)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("---------------------------------------------------------")

    # Train each model
    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

    # Extract feature importances or coefficients
    feature_importances = {}
    for name, pipeline in pipelines.items():
        if 'Logistic Regression' in name:
            feature_importances[name] = pipeline.named_steps['logisticregression'].coef_[0]

    df_test = prerequisites(testset_path, model_name)
    columns_to_encode = df_test.columns.tolist()
    X_test = df_test[columns_to_encode]
    y_test = df_test[result_column]

    X_train = df3[columns_to_encode]
    y_train = df3[result_column]

    return accuracy1

def make_training_directory(directory_name):

    # Specify the path where you want to create the directory
    directory_path = os.path.join(os.getcwd(), directory_name)

    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_name}' created successfully.")
    else:
        print(f"Directory '{directory_name}' already exists.")

def train_model(trainset_path, result_column, testset_path, model_name):
    make_training_directory(model_name)
    df3 = prerequisites(trainset_path, model_name)
    columns_to_encode = df3.columns.tolist()
    columns_to_encode.remove(result_column)
    X = df3[columns_to_encode]
    y = df3[result_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    run_logistic_regression(df3, X_train, X_test, y_train, y_test, result_column, testset_path, model_name)

    log_reg_model = LogisticRegression(max_iter=1000)

    # Fit the model to the training data
    log_reg_model.fit(X_train, y_train)
    base_path = model_name+os.path.sep
    model_path = base_path+model_name+".joblib"
    # Save the trained model
    dump(log_reg_model, model_path)

    dump(df3, base_path+model_name+'_encoded_df.joblib')

    print("Model trained and saved successfully at:", model_path)

def verify_and_parse(value):
    try:
        # Try to convert to an integer first
        int_value = int(value)
        print(f"'{value}' is an integer.")
        return int_value
    except ValueError:
        try:
            # If integer conversion fails, try to convert to a float
            float_value = float(value)
            print(f"'{value}' is a float.")
            return float_value
        except ValueError:
            # If both conversions fail, return the original string
            print(f"'{value}' is neither an integer nor a float.")
            return value

def get_mapping_real_value(df_train_set, df_encoded, column, encoded_value):
    if check_if_string(df_train_set[column]):
        print(f"\nMapping for column '{column}':")
        unique_values = df_train_set[column].unique().tolist()
        unique_values_encoded = df_encoded[column].unique().tolist()
        if isinstance(encoded_value, (str, object)):
            encoded_value = verify_and_parse(encoded_value)
        if encoded_value in unique_values_encoded:
            index = unique_values_encoded.index(encoded_value)
            return unique_values[index]
        else:
            return encoded_value
    else:
        return encoded_value



def test_model(model_name, single_data_point1, result_column):
    loaded_model = None
    base_path = model_name+os.path.sep

    try:
        loaded_model = load(base_path+model_name+'.joblib')
    except FileNotFoundError:
        print("Model file '"+model_name+".joblib' not found. Exiting test process.")
        return
    if loaded_model:
        single_data_point = json.loads(json.dumps(single_data_point1))

        # Convert to DataFrame with a single row
        single_df = pd.DataFrame([single_data_point])

        single_df = encode_columns(single_df)

        columns_to_encode = single_df.columns.tolist()
        if result_column in columns_to_encode:
            columns_to_encode.remove(result_column)


        # Select features
        X_single = single_df[columns_to_encode]

        prediction_single = loaded_model.predict(X_single)

        # Display prediction
        print("Prediction for the single data point using the loaded model:")

        decision = get_mapping_real_value(load(f"{model_name}/{model_name}_original_cleaned_df.joblib"), load(f"{model_name}/{model_name}_encoded_df.joblib"), result_column, prediction_single[0])
        print(decision)

import re

def is_numeric(s):
    pattern = re.compile(r'^-?\d+(\.\d+)?$')
    return bool(pattern.match(s))

def check_column_if_numeric(df):
    for column in df:
        if pd.api.types.is_numeric_dtype(column):
            return True
        else:
            # Check if the column contains numeric values stored as strings
            if is_numeric(column):
                return True
            else:
                return False

def convert_into_numeric_values(single_data_point_test_1):
    for key, value in single_data_point_test_1.items():
        if isinstance(value, (int, float)):
            single_data_point_test_1[key] = str(value)
    return single_data_point_test_1


