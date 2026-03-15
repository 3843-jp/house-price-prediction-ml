import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
data = pd.read_csv("kc_house_data.csv")

# Clear X parameters
X = data[['bedrooms','bathrooms','sqft_living','sqft_lot',
          'floors','waterfront','view','condition',
          'grade','sqft_above','sqft_basement','yr_built']]

y = data['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Polynomial features (degree 2 recommended)
"""poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)"""

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Price")
plt.show()
import pickle

filename1 = 'house_model.pkl'
with open(filename1, 'wb') as f:
    pickle.dump(model, f)

"""filename2 = 'poly.pkl'
with open(filename2, 'wb') as f:
    pickle.dump(poly, f)"""

print("Model  saved successfully")