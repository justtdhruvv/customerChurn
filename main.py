import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset into a pandas DataFrame
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# See the first 5 rows to get a feel for the data
print(df.head())

# Check column names, data types, and if there are any missing values
print(df.info())

# Get some statistical summaries of the numerical columns
print(df.describe())

sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Drop the customerID column
df = df.drop('customerID', axis=1)

# The column 'TotalCharges' might have some missing values if a customer is brand new
# We will fill these with 0
# First, convert to a numeric type, forcing errors (like empty strings) into NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Now, fill any NaNs that were created
df.fillna({'TotalCharges': 0}, inplace=True)

# Convert columns with 'Yes'/'No' to 1/0
# There are better ways, but this is the most straightforward!
for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert other categorical columns into numerical ones using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))