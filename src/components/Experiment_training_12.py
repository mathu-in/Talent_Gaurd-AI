import mlflow
import mlflow.sklearn
import sklearn

Experiment_name = 'Employee Attrition Prediction'

def setup_exp():
    mlflow.set_experiment(Experiment_name)

def log_exp(model, model_name, metrics):
    setup_exp()
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({'model name': model_name})
        model_params = model.get_params()

        for parameter_name , parameter_value in model_params.items():
            mlflow.log_param(parameter_name, parameter_value)

        for metric_name , metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # mlflow.sklearn.log_model(sk_model=model, name=model_name)
        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        print(model_name)
        print(Experiment_name)
        print("12. Experiment tracking is completed --- ML Flow")


def main():
    print("Run log_exp from main")