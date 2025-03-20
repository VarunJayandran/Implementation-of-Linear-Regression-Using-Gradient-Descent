# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters – Set initial values for slope m and intercept 𝑏 and choose a learning rate 𝛼
2. Compute Cost Function – Calculate the Mean Squared Error (MSE) to measure model performance.
3. Update Parameters Using Gradient Descent – Compute gradients and update m and b using the learning rate.
4. Repeat Until Convergence – Iterate until the cost function stabilizes or a maximum number of iterations is reached.

## Program:
```PYTHON
/*
Program to implement the linear regression using gradient descent.
Developed by: VARUN J C 
RegisterNumber:  212224240179
*/
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        
        errors=(predictions-y).reshape(-1,1)

        theta -=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("/content/50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2 , 136897.8 , 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![Screenshot 2025-03-20 082839](https://github.com/user-attachments/assets/bdfaf8e2-550c-4b8e-817d-f08a69f78e1d)


![Screenshot 2025-03-20 083058](https://github.com/user-attachments/assets/a6446255-8ad1-4691-a613-47c2f61c609d)
![Screenshot 2025-03-20 083113](https://github.com/user-attachments/assets/58ee41a7-c6c3-4f3e-a66f-b62eb95ab3b4)


![Screenshot 2025-03-20 083721](https://github.com/user-attachments/assets/48db31c3-62f3-4f2b-9a60-d5aa5c6d70e2)
![Screenshot 2025-03-20 083739](https://github.com/user-attachments/assets/34af5341-bec4-4880-beaf-655561dc8ec8)


![Screenshot 2025-03-20 083905](https://github.com/user-attachments/assets/6bc9b9ae-ace9-4d75-827a-4e07214165d4)





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
