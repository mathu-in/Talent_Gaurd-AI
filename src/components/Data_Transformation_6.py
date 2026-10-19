import pandas as pd


def classify_features(df: pd.DataFrame):
    num_features = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
    num_features.remove('EmployeeNumber') #2 line execution
    cat_features = df.select_dtypes(include=['object']).columns.to_list()
    cat_features.remove('Attrition')    #cat_features is converted to list already so no need to provide attrition in df.
    target_feature = 'Attrition'

    print("6. Data transformation done  --- Splitting Data into numerical/categorical features & target variable")
    return num_features,cat_features,target_feature

def main():
    from components.Data_ingestion_1 import load_data
    df = load_data()
    classify_features(df)

if __name__ == '__main__':
    main()