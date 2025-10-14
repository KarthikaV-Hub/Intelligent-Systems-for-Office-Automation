#Predicting temperature, precipitation, and other weather metrics based on meteorological data

import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


X = np.array([[60,2.0,12,1012],[65,0.5,10,1015],[70,1.2,8,1010],[55,3.0,15,1018],[75,0.0,5,1011]])
y = np.array([15.5,16.0,16.2,14.8,17.0])

model = Ridge(alpha=1.0).fit(X, y)
y_pred = model.predict(X)

print("MAE:", mean_absolute_error(y, y_pred))
print("R² Score:", r2_score(y, y_pred))

new = np.array([[60,2.0,12,1012]])
print("Predicted Temp (°C):", model.predict(new)[0])


plt.scatter(y, y_pred, color="blue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual Temp"); plt.ylabel("Predicted Temp")
plt.title("Actual vs Predicted Temperature")
plt.show()
     
