import torch

from helper import polynomial_feature_expansion, total_number_of_polynomial_terms, generate_weight, weighted_sum, sigmoid

from loss_functions.cross_entropy import MyCrossEntropy
from loss_functions.root_mean_square import MyRootMeanSquare

def logistic_fun(w, M, x):
    # logistic regression calculate the possibility of the output being 1
    #
    # Examples:
    # EX1: logistic_fun(torch.tensor([0.2, 2.2, 1.2, 4.3, 1.1, 0.4]), 2, torch.tensor([1, 2])) -> 0.76 (not an actual output)
    #
    # Parameters:
    # w: tensor containing the weights
    # M: integer representing the maximum polynomial order to consider
    # x: tensor containing the input features
    #
    # Returns:
    # float containing the possibility of the output being 1
    #
    # Usage:
    # This function is used to calculate the possibility of the output being 1

    #step1: generate polynomial and iteraction terms (Example: [1, x1, x2, x1^2, x1*x2, x2^2])
    generated_polynomial_features = polynomial_feature_expansion(x,M)

    #step2: calculate the weighted sum (Example: w0*1 + w1*x1 + w2*x2 + w4*x1^2 + w5*x1*x2 + w6*x2^2)
    weighted_sum_result = weighted_sum(w, generated_polynomial_features)

    #step3: calculate the possibility of the output being 1
    possibility = sigmoid(weighted_sum_result)

    return possibility

def fit_logistic_sgd(x, t, loss_fun="CROSS-ENTROPY", learning_rate=0.01, minibatch_size=20, M=2, num_iterations=5000):
    # Fit the logistic regression model using mini batch stochastic gradient descent
    #
    # Examples:
    # EX1: fit_logistic_sgd(torch.tensor([[1, 2], [2, 3]]), torch.tensor([0, 1]), "CrossEntropy", 0.01) -> tensor([0.2, 2.2, 1.2, 4.3, 1.1, 0.4]) (not an actual output)
    #
    # Parameters:
    # x: tensor containing the input features, for example: [[1, 2], [2, 3], [3, 4]]
    # t: tensor containing the target labels, for example: [0, 1, 0]
    # loss_fun: string representing the loss function to use (CrossEntropy or RootMeanSquare)
    # learning_rate: float representing the learning rate
    # minibatch_size: integer representing the number of samples to consider in each iteration
    # M: integer representing the maximum polynomial order to consider
    #
    # Returns:
    # tensor containing the optimal weights
    #
    # Usage:
    # This function is used to fit the logistic regression model using mini batch stochastic gradient descent

    P = total_number_of_polynomial_terms(x.shape[1], M)
    w = torch.randn(P, requires_grad=True)

    optimizer = torch.optim.SGD([w], lr=learning_rate)

    num_samples = x.shape[0]

    for iteration in range(num_iterations):

        if loss_fun == "CROSS-ENTROPY":
            loss_function = MyCrossEntropy()
        elif loss_fun == "ROOT-MEAN-SQUARE":
            loss_function = MyRootMeanSquare()
        else:
            raise ValueError(f"Unknown loss function: {loss_fun}")

        minibatch = torch.randperm(num_samples)[:minibatch_size]

        for i in minibatch:
            input = x[i]
            actual_target = t[i]
            predict_target = logistic_fun(w, M, input)
            loss_function.add(predict_target, actual_target)
        
        loss = loss_function.calculate_average_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration == 0:
            print(f"Initial Loss = {loss.item()}\n")
            print(f"{'Iteration':<15}{'Loss':<10}")
            print("-------------------------")
        elif iteration == num_iterations - 1:
            print("-------------------------\n")
            print(f"*** Final Loss = {loss.item()}\n")
        elif iteration % (num_iterations / 10) == 0:
            print(f"[Iteration {iteration}] Loss = {loss.item()}")

    return w

def generate_data(num_samples, num_features):
    # Generate dataset which contain input data and target labels
    # Use for generate train and test dataset
    #
    # Parameters:
    # num_samples: integer representing the number of samples to generate
    # num_features: integer representing the number of input features
    # num_polynomial_terms: integer representing the maximum polynomial order to consider
    #
    # Returns:
    # tuple containing input data and target labels (with and without noise)
    # 
    #
    # Usage:
    # generate input data and target labels for training and testing

    M = 2

    inputs = torch.FloatTensor(num_samples, num_features).uniform_(-5, 5)

    p = total_number_of_polynomial_terms(num_features, M)
    w = generate_weight(p) # based on pattern from the coursework instruction

    noisy_targets = torch.zeros(num_samples)
    true_targets = torch.zeros(num_samples)

    # Generate the target
    for i in range(num_samples):
        x = inputs[i]
        probability = logistic_fun(w, M, x)
        gaussian_noise = torch.randn(1)
        noisy_probability = gaussian_noise + probability

        noisy_targets[i] = 1 if noisy_probability >= 0.5 else 0
        true_targets[i] = 1 if probability >= 0.5 else 0

    # true_target is the target without any gaussian_noise use for model evaluation
    # noisy_targets is the target which add gaussian_noise use for model training
    return inputs, noisy_targets, true_targets


def evaluate_model(inputs, targets, M, w):
    # Evaluate the logistic regression model using F1 score
    #
    # Examples:
    # EX1: evaluate_model(torch.tensor([[1, 2], [2, 3]]), torch.tensor([0, 1]), 2, torch.tensor([0.2, 2.2, 1.2, 4.3, 1.1, 0.4])) -> 0.76 (not an actual output)
    #
    # Parameters:
    # inputs: tensor containing the input features
    # targets: tensor containing the target labels
    # M: integer representing the maximum polynomial order to consider
    # w: tensor containing the weights
    #
    # Returns:
    # float containing the F1 score of the model
    #
    # Usage:
    # This function is used to evaluate the logistic regression model

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(inputs.shape[0]):
        predicted_target = 1 if logistic_fun(w, M, inputs[i]) >= 0.5 else 0

        if predicted_target == 1 and targets[i] == 1:
            true_positives += 1
        elif predicted_target == 0 and targets[i] == 1:
            false_negatives += 1
        elif predicted_target == 1 and targets[i] == 0:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    if precision + recall == 0:
        return 0

    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

def main():
    # DATA GENERATION
    inputs, noisy_targets, true_targets = generate_data(300, 5)

    # DATA SEGMENTATION
    train_inputs, train_noisy_targets, true_targets_for_train_set = inputs[:200], noisy_targets[:200], true_targets[:200]
    unseen_inputs, unseen_noisy_targets, true_targets_for_unseen_set = inputs[200:], noisy_targets[200:], true_targets[200:]

    # PRINT HEADER
    print("=====================================\n")
    print("COURSEWORK 1 TASK 1\n".center(40))
    print("=====================================\n")

    def train_and_evaluate(loss_name):
        section = 1 if loss_name == "CROSS-ENTROPY" else 2

        print(f"SECTION {section}: {loss_name} LOSS\n".center(40))
        print("-------------------------------------\n")

        for i in range(1, 4):
            M = i

            print(f"[{section}.{i}] {loss_name} LOSS with M = {M}\n".center(40))
            print("********************\n".center(40))

            # TRAINING
            optimal_weights = fit_logistic_sgd(train_inputs, train_noisy_targets, M=M, loss_fun=loss_name)
            print(f"Optimal weights: {optimal_weights.detach()}\n")

            # EVALUATING
            print(f"> F1-Score on unseen data using true labels (labels without noise) as ground-truth: {evaluate_model(unseen_inputs, true_targets_for_unseen_set, M, optimal_weights) * 100}%\n")
            print(f"> F1-Score on training data using true labels (labels without noise) as ground-truth: {evaluate_model(train_inputs, true_targets_for_train_set, M, optimal_weights) * 100}%\n")

            print("========== OPTIONAL ==========\n")

            print(f"> F1-Score on unseen data using noisy labels as ground-truth: {evaluate_model(unseen_inputs, unseen_noisy_targets, M, optimal_weights) * 100}%\n")
            print(f"> F1-Score on training data using noisy labels as ground-truth: {evaluate_model(train_inputs, train_noisy_targets, M, optimal_weights) * 100}%\n")

            print(f"----------- END OF [{section}.{i}] ----------- \n")

    # TRAINING AND EVALUATING
    train_and_evaluate("CROSS-ENTROPY")
    train_and_evaluate("ROOT-MEAN-SQUARE")

    # DISCUSSION
    print("SECTION 3: DISCUSSION\n".center(40))
    print("-------------------------------------\n")
    print("[3.1] WHAT A METRIC (OTHER THAN LOSSES) IS APPROPRIATE FOR THIS CLASSIFICATION PROBLEM? (50 WORDS)\n")
    print(">> CHOOSING F1-SCORE OVER ACCURACY, AS IT'S BETTER FOR IMBALANCED DATASET (PROPORTION OF 0 AND 1 LABELS NOT SIMILAR).")
    print(">> WHILE OUR DATASET MAY NOT SIGNIFICANTLY IMBALANCED, REAL-WORLD DATA OFTEN IS. I WANT TO TREAT THIS COURSEWORK AS A")
    print(">> REAL-LIFE SCENARIO TO GAIN PRACTICE AND HANDS-ON EXPERIENCE WITH REALISTIC EVALUATION METRICS.\n")
    print(f"----------- END OF [3.1] ----------- \n")
    print("[3.2] COMMENT BRIEFLY ON THE METRIC REPORTS (100 WORDS)\n")

    print(">> F1-SCORES ARE HIGHEST WHEN THE HYPERPARAMETER M IS 2, AS IT ALIGNS WITH DATA GENERATION PROCESS.\n")

    print(">> THIS TREND HOLDS FOR BOTH LOSS FUNCTIONS(SEE 1.2 AND 2.2).\n")

    print(">> SINCE THE MODEL WAS TRAINED ON NOISY DATA, F1-SCORES ARE NATURALLY HIGHER WHEN EVALUATED AGAINST NOISY LABELS.\n")

    print(">> AND OBSERVED DATA USUALLY YIELDS HIGHER SCORES THAN UNSEEN.\n")

    print(">> THE MODEL ALSO PERFORMS WELL WITH UNDERLYING TRUE CLASS LABELS(SEE 1.2), SUGGESTING IT CAPTURES PATTERNS, NOT JUST MEMORIZING.\n")

    print(">> WHILE RMS YIELDS LOWER LOSS, CROSS-ENTROPY CONSISTENTLY PRODUCES BETTER F1-SCORES.\n")

    print(">> THIS IS BECAUSE THE LOGISTIC FUNCTION’S S-CURVE BEHAVIOR (NON-CONSTANT CHANGE RATE) MAKES RMS UNSUITABLE FOR THIS TASK.\n")

    print(f"----------- END OF [3.2] ----------- \n")

if __name__ == "__main__":
    main()
