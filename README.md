# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Collect the dataset consisting of the independent variable Hours Studied (X) and the dependent variable Marks Scored (Y). 
2. Divide the dataset into training and testing sets to build and test the model.
3. Train the model using the simple linear regression equation:
                  Y=b0 + b1X 
   where,
   b0=Intercept
   b1=slope(regression coefficient)
   X=hours studied
   Y=predicted marks
4.Predict the marks for given study hours using the trained model and evaluate performance using Mean Squared Error (MSE) and R² score. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DIVYA K
RegisterNumber: 25019198
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

# Display dataset
print("Dataset:\n", df.head())
df
X = df[["Hours_Studied"]]   # Independent variable
y = df["Marks_Scored"]      # Dependent variable

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")
*/
```

## Output:
![simple linear regression model for predicting the marks scored]<img width="548" height="352" alt="image" src="https://github.com/user-attachments/assets/398a6e86-5272-450d-949c-821bafd5fbbe" />

<img width="1060" height="718" alt="image" src="https://github.com/user-attachments/assets/2fcabcac-9ee1-4702-979e-ff3d3c93ef2f" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
