from components.Data_ingestion_1 import load_data
from components.Data_Validation_2 import dataset_validation
from components.Data_Analysis_EDA_3 import general_eda
from components.Data_Visualization_4 import create_visual
from components.Feature_Selection_5 import feature_selection
from components.Data_Transformation_6 import classify_features
from components.Train_test_split_7 import split_data
from components.Data_Preprocessing_8 import preprocess_data,encode_target
from components.Model_training_9 import train_model


def main():
    df = load_data()
    dataset_validation(df)
    general_eda(df)
    create_visual(df)
    X, y = feature_selection(df)
    n, c, t = classify_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_preprocessed, X_test_preprocessed, pre_process_nc_data = preprocess_data(X_train, X_test, n, c)
    y_train_preprocessed, y_test_preprocessed = encode_target(y_train, y_test)

    leaderBoard, bestModel, bestModelName =  train_model(X_train_preprocessed, y_train_preprocessed,
                                                         X_test_preprocessed, y_test_preprocessed)






if __name__ == "__main__":
    main()