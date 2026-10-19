import pandas as pd
#column transformation for each column
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from config.path import DATA_DIR, MODEL_DIR
import joblib

# from src.components.train_test_slit import split_data

def create_numeric_pipeline():
    return Pipeline(steps=[('numeric', StandardScaler())])

def create_categorical_pipeline():
    return Pipeline(steps=[('categorical', OneHotEncoder())])

def create_preprocessor(numeric_feature_names,category_feature_names):
    preprocessor = ColumnTransformer(transformers=[('numerical',create_numeric_pipeline(),numeric_feature_names),
                                                 ('categorical',create_categorical_pipeline(),category_feature_names)])
    return preprocessor


def preprocess_data(X_train, X_test, numerical_feature_names,category_feature_names):
    preprocessor = create_preprocessor(numerical_feature_names,category_feature_names)
    X_train_preprocessor = preprocessor.fit_transform(X_train)
    X_test_preprocessor = preprocessor.transform(X_test)
    joblib.dump(X_train_preprocessor, MODEL_DIR/'preprocessor.pkl')
    print("8. Data preprocessor saved --- preprocessor.pkl is saved that have X_train preprocessed data")
    print("a. Numerical and Categorical data are pre-processed")

    return X_train_preprocessor,X_test_preprocessor, preprocessor


def encode_target(y_train,y_test):
    mapping = {'No':0,'Yes':1}
    y_train_encoded = y_train.map(mapping)
    y_test_encoded = y_test.map(mapping)
    print("b. Target variabled are encoded No = 1, Yes = 0")

    return y_train_encoded,y_test_encoded


