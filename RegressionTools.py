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
    Ht = data[input_features[0]]    
    for i in range(1, len(input_features)):
        Ht = np.vstack((Ht, data[input_features[i]]))
    y = data[output_feature]
    HtH = Ht.dot(Ht.transpose())
    rss = np.linalg.inv(HtH).dot(Ht).dot(y)
    return rss
    
def get_regression_predictions(input_data, intercept, slope):
    return slope * input_data + intercept

def inverse_regression_predictions(output_data, intercept, slope):
    return (output_data - intercept) / slope

def get_residual_sum_of_squares(data, input_feature, output_feature, intercept, slope):
    x = data[input_feature]
    y = data[output_feature]
    rss = (y - (slope * x + intercept)) ** 2
    return rss.sum()

def get_residual_sum_of_squares_multiple_models(data, input_features, output_feature, coeffs):
    Ht = data[input_features[0]]    
    for i in range(1, len(input_features)):
        Ht = np.vstack((Ht, data[input_features[i]]))
    y = data[output_feature]
    return (y-coeffs.dot(Ht)).dot(y-coeffs.dot(Ht))
