from components.Data_ingestion_1 import load_data
from components.Data_Validation_2 import dataset_validation


def main():
    df = load_data()
    dataset_validation(df)






if __name__ == "__main__":
    main()