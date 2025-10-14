#House_price_prediction

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "Size": [1000,1500,2000,2500,3000],
    "Bedrooms": [2,3,3,4,4],
    "Age": [10,5,8,4,2],
    "Price": [200,300,400,450,500]
})

#Linear Regression

slr = LinearRegression().fit(df[["Size"]], df["Price"])
print("SLR Prediction (2200 sq.ft):", slr.predict([[2200]])[0])
plt.scatter(df["Size"], df["Price"], c="blue")
plt.plot(df["Size"], slr.predict(df[["Size"]]), c="red")
plt.title("Simple Linear Regression")
plt.show()


#multiple linear regression

mlr = LinearRegression().fit(df[["Size","Bedrooms","Age"]], df["Price"])
print("MLR Prediction (2200 sq.ft, 3 BR, 5 yrs):", mlr.predict([[2200,3,5]])[0])
plt.scatter(df.index, df["Price"], c="blue", label="Actual")
plt.scatter(df.index, mlr.predict(df[["Size","Bedrooms","Age"]]), c="red", label="Predicted")
plt.title("Multiple Linear Regression")
plt.legend()
plt.show()
