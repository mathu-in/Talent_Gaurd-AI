#finding the best model
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from components.Model_Factory import get_model
from config.path import MODEL_DIR, REPORTS_DIR


def evaluate_model(model, X_test, y_test):
    #model load and predict function.
    predictions = model.predict(X_test)
    #returns in dictionary - 4 values
    #used in train_model() to find metrics after fitting
    return {'Accuracy score': accuracy_score(y_test,predictions),
            'Precision score': precision_score(y_test,predictions,zero_division=0),
            'Recall score': recall_score(y_test,predictions),
            'F1 score': f1_score(y_test,predictions)}

def train_model(X_train,y_train,X_test,y_test):
    models = get_model()
    leaderBoard = []
    bestModel = None
    bestModelName = ''
    best_f1 = 0
    print("Training starts here...")
    for model_name, model_algorithm in models.items():
        print("*"*100)
        print("Training", model_name)
        model_algorithm.fit(X_train, y_train)
        metrics = evaluate_model(model_algorithm, X_test, y_test)
        #results of metrics saved in leaderboard
        leaderBoard.append({'model':model_algorithm,
                        'accuracy':round(metrics['Accuracy score'],4),
                        'precision':round(metrics['Precision score'],4),
                        'recall':round(metrics['Recall score'],4),
                        'f1 score':round(metrics['F1 score'],4)})

        print("leaderBoard After Appending")
        print(leaderBoard)
        if metrics['F1 score'] > best_f1:
            best_f1 = metrics['F1 score']
            bestModel = model_algorithm
            bestModelName = model_name
        print(f'{model_name} best model is {bestModel} with F1 score {best_f1}')

    leaderBoard = pd.DataFrame(leaderBoard)
    leaderBoard = leaderBoard.sort_values(by='f1 score',ascending=False)
    leaderBoard.reset_index(drop=True,inplace=True)
    print("9. Finding the best model")
    # print(f'a. Leaderboard after finding the best model is {leaderBoard}')
    print(f'b. Find the best model is {bestModel}')
    print(f'c. Find the best model by model name is {bestModelName}')

    return leaderBoard, bestModel, bestModelName

def save_best_model(model, model_name):
    joblib.dump(model, MODEL_DIR/f'{model_name}.pkl')


def model_training_pipeline(X_train, y_train, X_test, y_test):
    leaderBoard, bestModel, bestModelName = train_model(X_train, y_train, X_test, y_test)
    print("*"*100)
    print("leaderboard after training")
    print(leaderBoard)
    print("*"*100)
    save_best_model(bestModel, bestModelName)
    print("model saved")

    return leaderBoard, bestModel, bestModelName
