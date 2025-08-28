import torch

# built-in library
import itertools
import math

def polynomial_feature_expansion(input_vector, polynomial_order):
    # Generate all terms (first-higher order polynomial terms and interaction terms)
    # 
    # Examples:
    # EX1: input_vector = [x1, x2] and polynomial_order = 2, the output will be [1, x1, x2, x1^2, x1*x2, x2^2]
    # EX2: More concrete example, input_vector = [2, 3] and polynomial_order = 2, the output will be [1, 2, 3, 4, 6, 9]
    # 
    # Parameters:
    # input_vector: tensor containing the input features, example: [2,3]
    # polynomial_order: integer representing the maximum polynomial order to consider
    #
    # Returns:
    # tensor containing values of all polynomial and interaction terms
    #
    # Usage:
    # ready to be passed to the weighted_sum function

    features = [torch.ones(1)] # Start with the bias term (w0)

    # Add polynomial and interaction terms
    for order in range(1, polynomial_order + 1):
        for combination in itertools.combinations_with_replacement(range(len(input_vector)), order):
            feature = torch.prod(input_vector[list(combination)])
            features.append(feature)
    
    return torch.tensor(features)

def weighted_sum(weight_vector, polynomial_features):
    # Calculate the weighted sum
    #
    # Examples:
    # EX1: weight_vector = [w0, w1, w2, w3, w4, w5] and  polynomial_features = [1, x1, x2, x1^2, x1*x2, x2^2], the output = w0 + w1x1 + w2x2 + w3(x1^2) + w4(x1*x2) +w5(x2^2)
    # EX2: More concrete example: weight_vector = [2.3, 4.1, 3.5, 6.2, 9.4, 3.1] and polynomial_features = [1, 2, 3, 4, 6, 9], The output = 2.3 + 4.1*2 + 3.5*3 + 6.2*4 + 9.4*6 + 3.1*9 = 130.1
    #
    # Parameters:
    # weight_vector: tensor containing the weights
    # polynomial_features: tensor containing the polynomial and interaction terms (output of polynomial_feature_expansion function)
    #
    # Returns:
    # the weighted sum (float)
    #
    # Usage:
    # ready to be passed to the sigmoid function

    return torch.dot(weight_vector, polynomial_features)

def sigmoid(weighted_sum):
    return torch.sigmoid(weighted_sum)

def total_number_of_polynomial_terms(D, M):
    # Calculate the total number polynomial terms (first-higher polynomial terms + interaction terms)
    # based on this equation: 𝑝=Σ(𝐷+𝑚−1𝑚)𝑀𝑚=0
    #
    # Examples:
    # EX1: D = 2 and M = 2, the output will be 6
    # EX2: D = 3, M = 2, the output will be 10
    #
    # Parameters:
    # D: integer representing the number of input features
    # M: integer representing the maximum polynomial order to consider
    #
    # Returns:
    # integer representing the total number of polynomial terms (first-higher polynomial terms + interaction terms)
    #
    # Usage:
    # Sometimes it is necessary to know the total number of polynomial terms to initialize the correct number of weight terms
    # More concrete usage: use total number of polynomial terms to pass to some function which use to generate weights.

    total_features = 0

    for m in range(M + 1):
        total_features += math.comb(D + m - 1, m)

    return total_features

def generate_weight(p):
    # For the purpose of this coursework, we will need to generate the weights based on the following equation:
    # 𝐰 = [(−1)^𝒑*√𝑝/𝑝,(−1)^𝒑−𝟏*√𝑝−1/𝑝,(−1)^𝒑−𝟐*√𝑝−2/𝑝,…,−√3/𝑝,√2/𝑝,−1/𝑝]
    # and use it to generate test and train datasets
    #
    # Examples:
    # EX1: p = 2, the output will be tensor([-0.7071, 0.7071])
    # EX2: p = 3, the output will be tensor([-0.5774, -0.5774, 0.5774])
    #
    # Parameters:
    # p: integer representing the total number of polynomial terms (first-higher polynomial terms + interaction terms)
    #
    # Returns:
    # tensor of containing the weights
    #
    # Usage:
    # For the purpose of this coursework, we need to use these weights to generate the test and train datasets

    weights = torch.zeros(p)

    for index in range(p):
        weights[index] = (((-1) ** (p - index)) * (torch.sqrt(torch.tensor(p - index)))) / p

    return weights
