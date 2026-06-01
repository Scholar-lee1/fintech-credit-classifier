import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("train.csv")

# Define features by type for clean processing
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# Handling ALL missing values explicitly
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

for col in numerical_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Target variable: Mapping binary labels manually
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, "N": 0})

# Encoding text values & enforce numeric 1s and 0s
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# convert them to integers
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Separate features (X) and target (y)
drop_cols = [col for col in ['Loan_ID', 'Loan_Status'] if col in df.columns]
X = df.drop(columns=drop_cols)
y = df['Loan_Status']

# Split into Train (80%) and Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train baseline model 
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9. Make prediction
predictions = model.predict(X_test)

# 10. Print Confusion matrix and Classification report
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, predictions))
print("\n--- Classification Report ---")
print(classification_report(y_test, predictions))

# Saving trained model
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

