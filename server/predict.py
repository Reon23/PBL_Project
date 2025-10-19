import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --- Load CSV ---
df = pd.read_csv("ingredient_usage.csv")

# --- Features: menu items ---
X = df[["Pizza_Sold", "Pasta_Sold", "Burger_Sold"]]

# --- Targets: ingredients ---
y = df[["Cheese_Used", "Tomato_Used", "Dough_Used", "Lettuce_Used"]]

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train multi-output regressor ---
model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)

# --- Predict on test set ---
y_pred = model.predict(X_test)

# --- Evaluate ---
for i, col in enumerate(y.columns):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{col}: MAE={mae:.2f}, R²={r2:.2f}")

# --- Predict for a new day ---
sample_input = pd.DataFrame(
    {"Pizza_Sold": [60], "Pasta_Sold": [40], "Burger_Sold": [30]}
)

predicted_ingredients = model.predict(sample_input)
for ingredient, value in zip(y.columns, predicted_ingredients[0]):
    print(f"Predicted {ingredient}: {value:.2f}")
