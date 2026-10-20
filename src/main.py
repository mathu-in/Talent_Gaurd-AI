from components.Data_ingestion_1 import load_data
from components.Data_Validation_2 import dataset_validation
from components.Data_Analysis_EDA_3 import general_eda
from components.Data_Visualization_4 import create_visual
from components.Feature_Selection_5 import feature_selection
from components.Data_Transformation_6 import classify_features
from components.Train_test_split_7 import split_data
from components.Data_Preprocessing_8 import preprocess_data,encode_target
from components.Model_training_9 import train_model
from components.Model_Tuning_10 import tuning
from components.Model_Evaluation_11 import evaluate_model
from components.Experiment_training_12 import log_exp
from components.SMOTE_13 import apply_smote


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

    X_train_balanced, y_train_balanced = apply_smote(X_train_preprocessed, y_train_preprocessed)

    leaderBoard, bestModel, bestModelName =  train_model(X_train_balanced, y_train_balanced,
                                                         X_test_preprocessed, y_test_preprocessed)

    result = tuning(X_train_balanced , y_train_balanced)

    model = result['tuned_model']
    model_name = result['model_name']

    evaluation_results = evaluate_model(model, model_name, X_test_preprocessed, y_test_preprocessed)
    print(evaluation_results)

    # log_exp(model, model_name, evaluation_results)







if __name__ == "__main__":
    main()