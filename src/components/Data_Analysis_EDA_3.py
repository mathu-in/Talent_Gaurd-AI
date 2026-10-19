import pandas as pd
import os
from config.path import REPORTS_DIR

# artifacts_path = "../artifacts/reports"
#os.makedirs(artifacts_path, exist_ok=True)

def general_eda(df: pd.DataFrame):
    # print('\n'+'='*60)
    # print("Exploratory Data Analysis".center(60))
    # print('='*60)
    # print(f'\nDataset shape: {df.shape}')
    # print('\nTarget Variable Distribution')
    # print(df['Attrition'].value_counts())
    # print('\nNumerical Variable Distribution')
    # print(df.describe())
    # print('\nDepartment Variable Distribution')
    # print(df['Department'].value_counts())
    # print(pd.crosstab(df['Attrition'], df['OverTime']))
    print("3. General EDA report generated")

    # with open(os.path.join(artifacts_path, "report.txt"), "w") as text_file:
    with open(REPORTS_DIR/"Initial_Report", "w") as text_file:
        text_file.write("Report : EDA Analysis".center(20))
        text_file.write("\nExploratory Data Analysis")
        text_file.write('\n'+'=' * 60)
        text_file.write("\nShape of the dataset")
        text_file.write('\n')
        text_file.write(str(df.shape))
        text_file.write('\n' + '=' * 60)
        text_file.write('\nTarget Variable Distribution')
        text_file.write('\n')
        text_file.write('\n')
        text_file.write(str(df['Attrition'].value_counts()))
        text_file.write('\n' + '=' * 60)
        text_file.write('\nNumerical Variable Distribution')
        text_file.write('\n')
        text_file.write('\n')
        text_file.write(str(df.describe()))
        text_file.write('\n' + '=' * 60)
        text_file.write('\nDepartment Variable Distribution')
        text_file.write('\n')
        text_file.write('\n')
        text_file.write(str(df['Department'].value_counts()))
        text_file.write('\n' + '=' * 60)
        text_file.write("\nAttrition Versus Overtime")
        text_file.write('\n')
        text_file.write('\n')
        text_file.write(str(pd.crosstab(df['Attrition'], df['OverTime'])))
        text_file.write('\n' + '=' * 60)
        text_file.write('\n'+'*'*10+"End of the report"+'*'*10)

def main():
    from components.data_ingestion import load_data
    df = load_data()
    general_eda(df)

if __name__ == '__main__':
    main()

