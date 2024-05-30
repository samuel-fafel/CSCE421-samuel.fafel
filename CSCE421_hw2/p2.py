import numpy as np
from sklearn.linear_model import Ridge
np.random.seed(2024)


def generate_testing_data(num_data):
    # generata n=num_data datapoints
    # each data point has a one-dimension feature
    x = np.float32(np.random.randint(0, 100, num_data))/100
    y = np.sin(10*x)
    return x, y

def generate_training_data(num_data):
    # generata n=num_data datapoints
    # each data point has a one-dimension feature
    x = np.float32(np.random.randint(0, 100, num_data))/100
    y = np.sin(10*x) + np.random.rand(num_data)
    return x, y

def polynomial_feature(X, degree):
    # convert original features to polynomial features
    # output X_poly is of dimension n * degree,
    # where each row represents features of one data point
    poly_features = []
    for i in range(degree):
        poly_features.append(X**(i+1))
    X_poly = np.array(poly_features)
    X_poly=X_poly.T

    return X_poly

  
degree = 20
lmd = 0.001 # lambda
n_train = 10 # number of training data

x_test, y_test = generate_testing_data(num_data = 1000000)
X_test_poly = polynomial_feature(x_test, degree=degree)

x_train, y_train = generate_training_data(num_data = n_train)
X_train_poly = polynomial_feature(x_train, degree=degree)

model = Ridge(alpha = lmd, fit_intercept=True, solver="lsqr") 
model.fit(X_train_poly, y_train)
rmse_train = np.sqrt(np.mean(np.square(model.predict(X_train_poly) - y_train)))
print("Training RMSE: " + str(rmse_train))
rmse_test = np.sqrt(np.mean(np.square(model.predict(X_test_poly) - y_test)))
print("Testing RMSE: " + str(rmse_test))






