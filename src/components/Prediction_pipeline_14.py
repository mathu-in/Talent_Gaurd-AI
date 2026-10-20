import joblib
import pandas as pd
from  config.path import MODEL_DIR


class pred_pipe:
    def __init__(self):
        self.model = joblib.load(MODEL_DIR/'Logistic Regression.pkl')
        self.preprocessor = joblib.load(MODEL_DIR/'preprocessor.pkl')

    def predict(self, df:pd.DataFrame):
        transformed_df = self.preprocessor.transform(df)
        prediction = self.model.predict(transformed_df)[0]
        probability = self.model.predict_proba(transformed_df)[0]

        self.preprocessor = joblib.load(MODEL_DIR / 'preprocessor.pkl')
        print(type(self.preprocessor))
        print("Prediction pipeline executed")

        return prediction, probability

def main_predict():
    employee_data = {
        'Age': 32,
        'BusinessTravel': 'Travel_Rarely',
        'DailyRate': 1100,
        'Department': 'Research & Development',
        'DistanceFromHome': 10,
        'Education': 3,
        'EducationField': 'Medical',
        'EmployeeCount': 1,
        'EnvironmentSatisfaction': 3,
        'Gender': 'Male',
        'HourlyRate': 65,
        'JobInvolvement': 3,
        'JobLevel': 2,
        'JobRole': 'Research Scientist',
        'JobSatisfaction': 4,
        'MaritalStatus': 'Married',
        'MonthlyIncome': 5500,
        'MonthlyRate': 15000,
        'NumCompaniesWorked': 2,
        'Over18': 'Y',
        'OverTime': 'No',
        'PercentSalaryHike': 15,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'StandardHours': 80,
        'StockOptionLevel': 1,
        'TotalWorkingYears': 8,
        'TrainingTimesLastYear': 3,
        'WorkLifeBalance': 3,
        'YearsAtCompany': 5,
        'YearsInCurrentRole': 3,
        'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': 3
    }
    employee_df = pd.DataFrame([employee_data])
    pipe = pred_pipe()
    prediction, probability = pipe.predict(employee_df)
    if prediction == 1:
        print('Employee likely to leave')
        print(f'Probability: {probability[1]*100:.2f}%')

    else:
        print('Employee likely to stay')
        print(f'Probability: {probability[0]*100:.2f}%')