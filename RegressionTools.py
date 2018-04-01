import numpy as np

def simple_linear_regression(data, input_feature, output_feature):
    # closed-form solution
    x = data[input_feature]
    y = data[output_feature]
    N = len(x) * 1.0
    xy = x * y
    xsquared = x ** 2
    slope = (xy.sum() - (y.sum() * x.sum() / N)) / (xsquared.sum() - (x.sum() * x.sum() / N))
    intercept = y.sum() / N - slope * x.sum() / N
    return intercept, slope

def multiple_linear_regression(data, input_features, output_feature):
    y = np.array(data[output_feature])
    N = len(data[input_features[0]])
    H = data[input_features[0]].reshape(N,1)
    for i in range(1, len(input_features)):
        H = np.hstack((H, data[input_features[i]].reshape(N,1)))    
    return np.linalg.inv(H.T.dot(H)).dot(H.T).dot(y)

def get_regression_predictions(input_data, intercept, slope):
    return slope * input_data + intercept

def get_multiple_regression_predictions(data, input_features, weights):
    y = data[input_features[0]] * weights[0]
    for i in range(1, len(input_features)):
        y += data[input_features[i]] * weights[i]
    return y

def inverse_regression_predictions(output_data, intercept, slope):
    return (output_data - intercept) / slope

def get_residual_sum_of_squares(data, input_feature, output_feature, intercept, slope):
    x = data[input_feature]
    y = data[output_feature]
    rss = (y - (slope * x + intercept)) ** 2
    return rss.sum()

def get_residual_sum_of_squares_multiple_models(data, input_features, output_feature, coeffs):
    y = np.array(data[output_feature])
    N = len(data[input_features[0]])
    H = data[input_features[0]].reshape(N,1)
    for i in range(1, len(input_features)):
        H = np.hstack((H, data[input_features[i]].reshape(N,1)))
    return (y-H.dot(coeffs.T)).dot(y-H.dot(coeffs.T))
