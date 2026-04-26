import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Synthetic daily sales data
dates = pd.date_range('2023-01-01', periods=500, freq='D')
sales = 100 + 10 * np.sin(np.arange(500) * 2 * np.pi / 30) + np.random.normal(0, 5, 500)
df = pd.DataFrame({'date': dates, 'sales': sales})

# Feature engineering
df['day_of_year'] = df['date'].dt.dayofyear
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Split by index (first 450 days train, last 50 days test)
train = df.iloc[:450]
test = df.iloc[450:]

X_train = train[['day_of_year', 'day_of_week', 'month']]
y_train = train['sales']
X_test = test[['day_of_year', 'day_of_week', 'month']]
y_test = test['sales']

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
print(f"Mean Absolute Error: {mae:.2f}")

# Create a business‑friendly forecast plot
plt.figure(figsize=(12, 5))
plt.plot(test['date'], y_test, label='Actual Sales', marker='o')
plt.plot(test['date'], preds, label='Forecasted Sales', linestyle='--', marker='x')
plt.title('Sales & Demand Forecast (Next 50 Days)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('sales_forecast.png')
plt.show()

# Business insight
trend = "upward" if preds[-1] > preds[0] else "downward"
print(f"\n📈 Business Insight: Sales are trending {trend} over the forecast period.")
print(f"Average predicted daily sales: {preds.mean():.2f}")