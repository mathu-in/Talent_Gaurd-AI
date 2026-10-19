import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report, confusion_matrix, roc_curve, ConfusionMatrixDisplay

from config.path import CHART_DIR


def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    probability = model.predict_proba(X_test)[:,1]
    metrics = {'Accuracy': accuracy_score(y_test, y_pred),
               'Precision': precision_score(y_test, y_pred, zero_division=0),
               'Recall': recall_score(y_test, y_pred, zero_division=0),
               'F1 Score': f1_score(y_test, y_pred, zero_division=0),
               'ROC AUC': roc_auc_score(y_test, y_pred)}

    return y_pred, probability, metrics

def display_metrics(metrics):
    for metric_name, metric_value in metrics.items():
        print(f'{metric_name}: {metric_value:.2f}')


def display_classfication_metrics(y_test, y_pred):
    report = classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition'],zero_division=0)
    print(report)

def display_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Attrition', 'Attrition'])
    display.plot()
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'confusion_matrix.png')
    # plt.show()
    plt.close()

def display_roc_curve(y_test, probability, auc_score):
    # false_positive_rate , true_positive_rate, auc_score = roc_curve(y_test, probability)
    false_positive_rate , true_positive_rate, threshold = roc_curve(y_test, probability)
    plt.figure(figsize=(10,10))
    plt.plot(false_positive_rate, true_positive_rate, label=f'ROC-AUC={auc_score:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'roc_Before.png')
    # plt.show()
    plt.close()

def evaluate_model(model, model_name, X_test, y_test):
    print(f'model: {model_name}')
    y_pred, probability, metrics = calculate_metrics(model, X_test, y_test)
    display_metrics(metrics)
    display_classfication_metrics(y_test, y_pred)
    display_confusion_matrix(y_test, y_pred)
    display_roc_curve(y_test, probability, auc_score=metrics['ROC AUC'])

    return metrics
