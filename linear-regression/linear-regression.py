import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation, linear_model

data = np.genfromtxt('data-houses.csv',delimiter=',');
data = np.matrix(data);


x = data[:,0];
y = data[:,1];

m = len(data);

X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.4)

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, regr.predict(X_test), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()


