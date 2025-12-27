# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weights and bias, and set learning rate and iterations.
2. Compute predicted output using sigmoid function.
3. Calculate error and gradients using log loss.
4. Update weights and bias using gradient descent until convergence.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: dilip kumar R
RegisterNumber: 25017135
#Using scikit-learn SGDRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Features and target
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
])
y = np.array([50, 45, 70, 80, 95])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create SGD Regressor
sgd_reg = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01, random_state=42)
sgd_reg.fit(X_scaled, y)

# Coefficients and intercept
print("Weights (coefficients):", sgd_reg.coef_)
print("Intercept:", sgd_reg.intercept_)

# Predictions
y_pred = sgd_reg.predict(X_scaled)
print("Predicted values:", y_pred)
 
*/
```

## Output:
Weights (coefficients): [8.97060815 3.37269876 7.00639208]
Intercept: [67.73138084]
Predicted values: [49.9211893  44.06349921 70.77494368 80.16177255 93.73549945]


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

