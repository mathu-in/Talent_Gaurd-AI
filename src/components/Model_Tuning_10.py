import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from components.Model_Factory import get_model

models = {'Random Forest Classifier': get_model()['Random Forest'],
          'Gradient Boosting': get_model()['Gradient Boosting']}

parameter_grid = {
    'Random Forest Classifier': {'n_estimators': [100, 200],
                                 'max_depth': [1, 2, 3],
                                 'min_samples_split': [2, 3, 5]},
    'Gradient Boosting' : {'n_estimators': [100, 200],
                           'max_depth': [1, 2],
                           'learning_rate': [0.01,0.1, 0.2],}
}

def tuning(X_train,y_train):
    tuning_results = {}
    best_model_name = None
    best_model = None
    best_score = -1
    best_params = None
    print("Tuning Model")

    for model_name, model_alg in models.items():
        print(f'Tuning {model_name}')
        parameters = parameter_grid[model_name]
        grid_Search = GridSearchCV(estimator=model_alg, param_grid=parameters, cv=8, n_jobs=-1)
        grid_Search.fit(X_train,y_train)
        update_best_score = grid_Search.best_score_
        update_best_params = grid_Search.best_params_
        update_best_model = grid_Search.best_estimator_

        tuning_results[model_name] = {'best_model': update_best_model,
                                      'best_params': update_best_params,
                                      'best_score': update_best_score,
                                      'best_model_name': model_name}
                                                                ##this variable will have the data of models present in parameter_grid variable

        if update_best_score > best_score:
            best_score = update_best_score
            best_model = update_best_model
            best_params = update_best_params
            best_model_name = model_name                 ##which one have higher compared to other models will be shown here based on condition

    print(f"Best Model: {best_model}")
    print(f"Best Parameters: {best_params}")
    print(f"Best score: {best_score}")
    print(f'Best Model Name: {best_model_name}')
    print("10. tuning complete")
    print("*"*100)

    return {'tuned_model': best_model,
            'tuned_params': best_params,
            'tuned_score': best_score,
            'tuning_results': tuning_results,
            'model_name': best_model_name}

