# Importing the required libraries

from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Reading data from the remote link using requests
url = r"https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
response = requests.get(url, verify=False)

# Use StringIO to create a file-like object from the response text
s_data = pd.read_csv(StringIO(response.text))

print("Data import successful")
s_data.head(10)
# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')
plt.title('Hrs vs Pct(%)')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score(%)')
plt.show()
X = s_data.iloc[:, :-1].values
y = s_data.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)

print("Training complete.")
# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line, color='red')
plt.show()

# Testing data
print(X_test)
# Model Prediction
y_pred = regressor.predict(X_test)
# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
# Estimating training and test score
print("Training Score:", regressor.score(X_train, y_train))
print("Test Score:", regressor.score(X_test, y_test))
# Plotting the Bar graph to depict the difference between the actual and predicted value

df.plot(kind='bar', figsize=(5, 5))
plt.grid(which='major', linewidth='0.5', color='red')
plt.grid(which='minor', linewidth='0.5', color='blue')
plt.show()
# Testing the model with our own data
hours = 9.25
test = np.array([hours])
test = test.reshape(-1, 1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(
    metrics.mean_squared_error(y_test, y_pred)))
print('R-2:', metrics.r2_score(y_test, y_pred))
