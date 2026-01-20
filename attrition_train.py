# Just copied from Final_Exam.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder




from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


import pickle


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier




import warnings
warnings.filterwarnings("ignore")



# ===================================================

#                   Data Loading
# ===================================================

# ======================== Load the dataset ======================

# Stored the dataset from my personal google drive and make it public for the tasks
file_link = "https://drive.google.com/file/d/1O0-B52ueUlnCVpXPhjrT3UrWw4BY3iMY/view?usp=drive_link"

# get the id part of the file
id = file_link.split("/")[-2] 

# creating a new link using the id so that we can easily read the csv file in pandas
new_link = f'https://drive.google.com/uc?id={id}'

try:
  df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")   # Local Environment - Download the .csv file from Kaggle

except:
  #load the dataset from my google drive uploaded , If the upper one fails
  df = pd.read_csv(new_link)


# ================Display first 15 rows with shape ==================
print(df.head(15))


print("="*80)
print(df.shape)
print("="*80)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


print("="*80)
print("\nDescribe (numeric columns):")
print("\n" + "="*80)

# display(df.describe().T)
print(df.describe().T)




# ===================================================
#             Data Preprocessing
# ===================================================

# Data Preprocessing

# ============= Handling Missing Values =============
print("="*80)
print("Missing Values")
print("\n" + "="*80)

missing = df.isnull().sum()
print(missing)         # No Missing Value Found Here 



# ============= Finding Unique Values =============
print("="*80)
print("Unique Values")
print("\n" + "="*80)

print(df.nunique())


# ============= Drop Columns =============

# EmployeeCount, Over18, StandardHours theres only one unique value and it was same for everyone so that's not needed, dropping this columns
# EmployeeNumber is just an identifier and should not be used for prediction

cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']

df = df.drop(columns=cols_to_drop)
# print(df.shape)



# ===================== Encoding Categorical Target Variables================

le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])
print("Target variable encoded:")
print(df['Attrition'].value_counts())

# Separate X and y
X = df.drop('Attrition',axis=1)
y = df['Attrition']

# one Hot Label Encoding needed in --> BusinessTravel, Department, EducationField, JobRole , MaritalStatus
# Label Encoding --> Attrition, Gender, OverTime


# Numerical Column and Categorical Columns
numeric_features = X.select_dtypes(include = ['int64','float64']).columns
categorical_features = X.select_dtypes(include = ['object']).columns

print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ============ Check correlation for numerical features ==============
corr_target = df.select_dtypes(include=np.number).corr()['Attrition'].sort_values(ascending=False)
print("Correlation with Attrition:")
print(corr_target)



# ===================================================
#               Pipeline Creation
# ===================================================

# ======================== PIPELINE CREATION ==================


#for Numerical features
num_transformer = Pipeline (
    steps = [
        ('imputer', SimpleImputer(strategy= 'median')),
        ('scaler', StandardScaler())
    ]
)


# For categorical feature
cat_transformer = Pipeline( steps = [
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
] )


# split
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state=42)

#combine them
preprocessor = ColumnTransformer(
    transformers= [
        ('num',num_transformer, numeric_features),
        ('cat',cat_transformer, categorical_features)
    ]
    )



# ===================================================
#               Primary Model Selection
# ===================================================

#========================== Primary Model Selection====================

# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=1
)

# ML pipeline=
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

print("ML Pipeline Created --> Preprocessing + Random Forest")


# =================================================
#              Model Training
# =================================================

# Ensemble - boosting, stacking

#base learner
clf_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_rf = RandomForestClassifier( n_estimators=100, random_state=42 )
clf_gb = GradientBoostingClassifier( n_estimators=100 , random_state=42 )


#Voting Classifier
voting_clf = VotingClassifier(
    estimators= [
        ('lr', clf_lr),
        ('rf',clf_rf),
        ('gb', clf_gb)
    ]
)


#stacking
stacking_clf = StackingClassifier(
    estimators= [
        ('rf',clf_rf),
        ('gb', clf_gb)
    ],
    final_estimator= LogisticRegression(max_iter=1000) # the meta learner that shows on project week
)


# dictionary of all model
model_to_train = {
    'Logistic Regression' : clf_lr,
    'Random Forest' : clf_rf,
    'Gradient Boosting': clf_gb,
    'Voting Ensemble ' : voting_clf,
    'Stacking Ensemble ' : stacking_clf

}


# Training & Evaluation
result = []

# Creating full pipeline with preprocessor
for name , model in model_to_train.items():
  pipe = Pipeline(
      [
          ('preprocessor', preprocessor),
          ('model',model)
      ]
  )


  pipe.fit(X_train,y_train)            # train

  y_pred = pipe.predict(X_test)     #predict

  accuracy = accuracy_score(y_test,y_pred)   # Evaluate
  
  result.append({
      "Model": name,
      "Accuracy" :accuracy
  })

results_df = pd.DataFrame(result).sort_values("Accuracy", ascending=False)

print(results_df)





# =================================================
#              Cross Validation
# =================================================

rf_pipeline = Pipeline(
    [
        ('preprocessor',preprocessor),
        ('model',RandomForestClassifier(n_estimators=100,random_state=42))
     ]
  )

# ============ 5 fold crpss validation ===============
cv_scores = cross_val_score( rf_pipeline,X_train,y_train,cv=5, scoring='accuracy' )

print("Cross-Validation Scores:", cv_scores)
print(f"\nAverage CV Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")





# =================================================
#              Hyperparameter Tuning
# =================================================
from sklearn.model_selection import GridSearchCV

rf_pipeline = Pipeline(
    [
        ('preprocessor',preprocessor),
        ('model',RandomForestClassifier(random_state=42))
     ]
  )



# Define the grid
param_grid = {
    'model__n_estimators' : [100,200] ,
    'model__max_depth': [None,10,20],
    'model__min_samples_split' : [2,5]
}

grid_search = GridSearchCV(
    estimator = rf_pipeline,
    param_grid = param_grid,
    cv = 5 ,
    scoring = 'accuracy',
    n_jobs =1,
    verbose = 2

)


grid_search.fit(X_train,y_train)

print("Best Score:", grid_search.best_score_)
print("Best Parameters:", grid_search.best_params_)


# =================================================
#              Best Model Selection
# =================================================

# Extract the best model using GridSearchCV

best_rf = grid_search.best_estimator_
# print("Best Parameters:", grid_search.best_params_)

print("="*80)
print("Best Cross-Validation Score:")
print("="*80)
print(f"{grid_search.best_score_:.4f}")

# Assign to best_model
best_model = best_rf




# =================================================
#              Model Performance Evaluation
# =================================================

# ======== Evaluate the Model on test Set ===============

# Predict on test set
y_pred = best_model.predict(X_test)


#Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

print("Model Performance on Test Set:")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# =========== Visualization of Confusion Matrix ================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
# plt.show()
plt.savefig('confusion_matrix.png')
print("\n Confusion matrix saved as 'confusion_matrix.png'")



# =================================================
#              Save the Model
# =================================================
# Save the Model on Google Collab

filename = "attrition_rf_model_pipeline.pkl"

with open( filename, "wb" ) as file:
  pickle.dump( best_model, file )
print(f"\nModel saved as '{filename}'")