# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pickle import dump

# Load Dataset
data = pd.read_csv("cs8_car_price_feb26.csv")

print(data.head())
print(data.info())

# Check missing values
print(data.isnull().sum())

# ---------- Convert Categorical to Numerical ----------
ndata = pd.get_dummies(data, columns=["car_name"])

print(ndata.head())

# ---------- Features and Target ----------
features = ndata.drop(["price"], axis=1)
target = ndata["price"]

# ---------- Train Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=100)

# ---------- Model ----------
model = LinearRegression()
model.fit(X_train, y_train)

print("Model Training Completed")

# ---------- Prediction ----------
y_pred = model.predict(X_test)

# ---------- Model Evaluation ----------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

#print("\nModel Performance")
print("R2 Score :", r2)
print("MAE :", mae)
print("MSE :", mse)

# ---------- Visualization ----------
plt.scatter(data["kms_driven"], data["price"])
plt.xlabel("Kilometers Driven")
plt.ylabel("Car Price")
plt.title("Car Price vs Kilometers Driven")
plt.show()

# ---------- Save Model ----------
with open("car_price_model.pkl", "wb") as f:
	dump(model, f)

print("Model Saved Successfully")