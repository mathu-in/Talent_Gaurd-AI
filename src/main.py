from components.Data_ingestion_1 import load_data
from components.Data_Validation_2 import dataset_validation
from components.Data_Analysis_EDA_3 import general_eda

def main():
    df = load_data()
    dataset_validation(df)
    general_eda(df)






if __name__ == "__main__":
    main()