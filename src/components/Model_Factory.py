from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def get_model():
    models = {'Logistic Regression' : LogisticRegression(random_state=42, max_iter=1000),
              'Decision Tree' : DecisionTreeClassifier(random_state=42),
              'Random Forest' : RandomForestClassifier(random_state=42),
              'Gradient Boosting' : GradientBoostingClassifier(random_state=42),
              'AdaBoost' : AdaBoostClassifier(random_state=42),
              'k-Nearest Neighbors' : KNeighborsClassifier(n_neighbors=42, n_jobs=-1),
              'GaussianNB' : GaussianNB()}

    print("Created the models")
    print(models)

    # print("Available models:")
    # for model_name in models:
    #     print(model_name)

    return models

def main():
    models = get_model()
    print("Available models:")
    # for model_name, model in models.items():
    for model_name in models:
        print(model_name)


if __name__ == "__main__":
    main()