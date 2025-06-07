import inspect
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
#print(inspect.signature(LinearRegression))
#print(inspect.getdoc(LinearRegression))
#print(dir(LinearRegression))
#print(help(pd.DataFrame))
#LinearRegression.get_params()
import numpy as np

X=np.array([1,2,3,4,5]).reshape(-1,1)
y=np.array([7,2,5,20,1])
model=LinearRegression()
model.fit(X,y)
y_pred=model.predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Best Fit Line (Least Squares)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression - Least Squares Line')
plt.legend()
plt.show()