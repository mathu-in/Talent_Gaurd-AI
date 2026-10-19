import pandas as pd

def feature_selection(df: pd.DataFrame):
    # df = df.drop(columns=['Employee ID'],axis=1)

    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    print("X-Shape: ", X.shape)
    print("Y-Shape: ", y.shape)
    print("5. Feature Selection executed")

    return X, y


def main():
    from src.config.path import DATA_DIR
    # from src.components.Data_Validation_2 import load_data

    df = pd.read_csv(DATA_DIR/'EmployeeAttrition.csv')
    X, y = feature_selection(df)

    print("X-Shape: ", X.shape)
    print("Y-Shape: ", y.shape)
    print("Feature Selection Complete")



if __name__ == '__main__':
    main()