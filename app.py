#gradio app 
import gradio as gr
import pandas as pd
import pickle

# 1. Load the Model
with open("attrition_rf_model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# The Logic Function
def predict_attrition(Age, BusinessTravel, DailyRate, Department, DistanceFromHome,
                     Education, EducationField, EnvironmentSatisfaction, Gender,
                     HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction,
                     MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked,
                     OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction,
                     StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear,
                     WorkLifeBalance, YearsAtCompany, YearsInCurrentRole,
                     YearsSinceLastPromotion, YearsWithCurrManager):
    

    # Pack inputs into a DataFrame
    input_df = pd.DataFrame([[
        Age, BusinessTravel, DailyRate, Department, DistanceFromHome,
        Education, EducationField, EnvironmentSatisfaction, Gender,
        HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction,
        MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked,
        OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction,
        StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear,
        WorkLifeBalance, YearsAtCompany, YearsInCurrentRole,
        YearsSinceLastPromotion, YearsWithCurrManager
    ]],
      columns=[
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
        'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
        'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
        'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    # Return formatted result
    if prediction == 1:
        result = "Yes - Employee will leave"
    else:
        result = "No - Employee will stay"
    
    return f"Attrition Prediction: {result}"

# The App Interface
inputs = [
    gr.Number(label="Age", value=30),
    gr.Radio(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], label="Business Travel"),
    gr.Number(label="Daily Rate", value=800),
    gr.Radio(["Sales", "Research & Development", "Human Resources"], label="Department"),
    gr.Number(label="Distance From Home (km)", value=5),
    gr.Slider(1, 5, step=1, label="Education Level", value=3),
    gr.Dropdown(["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"], label="Education Field"),
    gr.Slider(1, 4, step=1, label="Environment Satisfaction", value=3),
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Number(label="Hourly Rate", value=65),
    gr.Slider(1, 4, step=1, label="Job Involvement", value=3),
    gr.Slider(1, 5, step=1, label="Job Level", value=2),
    gr.Dropdown(["Sales Executive", "Research Scientist", "Laboratory Technician", 
                 "Manufacturing Director", "Healthcare Representative", "Manager",
                 "Sales Representative", "Research Director", "Human Resources"], label="Job Role"),
    gr.Slider(1, 4, step=1, label="Job Satisfaction", value=3),
    gr.Radio(["Single", "Married", "Divorced"], label="Marital Status"),
    gr.Number(label="Monthly Income", value=5000),
    gr.Number(label="Monthly Rate", value=15000),
    gr.Number(label="Num Companies Worked", value=2),
    gr.Radio(["Yes", "No"], label="Over Time"),
    gr.Slider(11, 25, step=1, label="Percent Salary Hike", value=15),
    gr.Slider(3, 4, step=1, label="Performance Rating", value=3),
    gr.Slider(1, 4, step=1, label="Relationship Satisfaction", value=3),
    gr.Slider(0, 3, step=1, label="Stock Option Level", value=1),
    gr.Number(label="Total Working Years", value=8),
    gr.Slider(0, 6, step=1, label="Training Times Last Year", value=3),
    gr.Slider(1, 4, step=1, label="Work Life Balance", value=3),
    gr.Number(label="Years At Company", value=5),
    gr.Number(label="Years In Current Role", value=3),
    gr.Number(label="Years Since Last Promotion", value=1),
    gr.Number(label="Years With Current Manager", value=3)
]



app = gr.Interface(
    fn=predict_attrition,
    inputs=inputs,
    outputs="text", 
    title="Employee Attrition Predictor",
    description="Predict if an employee will leave the company"
)

app.launch(share=True)
