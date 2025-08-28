import torch

from task import generate_data, fit_logistic_sgd, evaluate_model

class MController(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # START WITH EQUAL PROBABILITY FOR ALL M
        # FOR THE PURPOSE OF THIS COURSEWORK, I SET M ONLY 3 POSSIBLE VALUES (1,2,3)
        self.logits = torch.nn.Parameter(torch.zeros(3))

    def forward(self):
        return torch.softmax(self.logits, dim=0)

def sample_M_index(probabilities):
    # Randomly pick an index based on given probabilities.
    #
    # Note that, the index with a higher probability has a higher chance of being selected, but it is not guaranteed.
    #
    # Examples:
    # sample_M_index(torch.tensor([0.1, 0.3, 0.6])) -> 2
    #
    # Parameters:
    # probabilities: A tensor containing the probabilities for each index.
    #
    # Returns:
    # The randomly selected index based on the given probabilities.
    #
    # Usages
    # Each iteration, we need to randomly select an index of M based on weighted probabilities.
    return torch.multinomial(probabilities, 1).item()

def index_to_M(index):
    # Convert an index to its corresponding M value.
    #
    # Examples:
    # index_to_M(0) -> 1
    # index_to_M(1) -> 2
    # index_to_M(2) -> 3
    #
    # Parameters:
    # index: An integer index.
    #
    # Returns:
    # The corresponding M value for the given index.
    #
    # Usages
    # Map the sampled index to its corresponding M value.

    # ADD MORE IF NEEDED
    match index:
        case 0:
            return 1
        case 1:
            return 2
        case 2:
            return 3

def fit_sgd(train_inputs, train_noisy_targets, unseen_inputs, true_targets_for_unseen_set, learning_rate=0.1, num_iterations=10):
    # Fit a model using SGD and reinforcement learning to find the optimal hyperparameter M.
    #
    # Example:
    # [ NOT actual output! ]
    # fit_sgd(tensor([0.1, 0.3, 0.6, 0.5, 0.2]), tensor([1]), tensor([0.3, 0.1, 0.4, 0.9, 0.1]), tensor([1])) -> M = 2, f1-score = 0.76
    # 
    # Parameters:
    # train_inputs: The input data for training.
    # train_noisy_targets: The noisy target labels for training.
    # unseen_inputs: The input data for evaluation.
    # true_targets_for_unseen_set: The true target labels for evaluation.
    # learning_rate: The learning rate for the SGD optimizer.
    # num_iterations: The number of iterations for the reinforcement learning process.
    #
    # Returns:
    # The optimal M value and the corresponding F1-score.
    #
    # Usages
    # Find the optimal hyperparameter M and it's f1-score.

    controller = MController()
    optimizer = torch.optim.SGD(controller.parameters(), lr=learning_rate)

    optimal_M = None
    optimal_f1 = 0.0

    print(f"INITIAL PROBABILITIES: {controller().detach()}\n")
    print("=====================================\n")

    # I THINK 10 ITERATIONS ARE USUALLY ENOUGH, BUT OF COURSE MORE ITERATIONS ALWAYS BETTER!
    for i in range(1, num_iterations + 1):
        probs = controller()

        M_index = sample_M_index(probs)
        M = index_to_M(M_index)

        print(f"ITERATION {i}/{num_iterations} TRYING M = {M}\n")

        # TRAIN THE MODEL WITH THE SELECTED M (FROM TASK 1 WE PROVED THAT CROSS-ENTROPY IS A BETTER LOSS FUNCTION)
        optimal_weight = fit_logistic_sgd(train_inputs, train_noisy_targets, M=M, loss_fun="CROSS-ENTROPY", num_iterations=1000)

        # USE F1-SCORE AS A REWARD (USE UNDERLYING TRUE CLASS AS A GROUND TRUTH JUST LIKE WE DID IN TASK1)
        f1_score = evaluate_model(unseen_inputs, true_targets_for_unseen_set, M, optimal_weight)

        print(f"*** F1-SCORE = {f1_score}\n")

        # IF F1-SCORE MORE THAN OPTIMAL ONE, LOSS WILL BE NEGATIVE -> WE DOING BETTER (GET REWARD) 
        # IF F1-SCORE LESS THAN OPTIMAL ONE, LOSS WILL BE POSITIVE -> WE DOING WORST (GET PUNISH)
        loss = -torch.log(probs[M_index]) * (f1_score - optimal_f1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"UPDATED PROBABILITIES: {controller().detach()}\n")

        if f1_score > optimal_f1:
            print(f"!!! ALERT !!! OPTIMAL F1-SCORE CHANGE FROM {optimal_f1} TO {f1_score}\n")
            optimal_f1 = f1_score
            optimal_M = M

        print("=====================================\n")
    
    return optimal_M, optimal_f1

def main():
    # DATA GENERATION
    inputs, noisy_targets, true_targets = generate_data(300, 5)

    # DATA SEGMENTATION
    train_inputs, train_noisy_targets, true_targets_for_train_set = inputs[:200], noisy_targets[:200], true_targets[:200]
    unseen_inputs, unseen_noisy_targets, true_targets_for_unseen_set = inputs[200:], noisy_targets[200:], true_targets[200:]

    # PRINT HEADER
    print("=====================================\n")
    print("COURSEWORK 1 TASK 1A\n".center(40))
    print("=====================================\n")

    print(">> SOLUTION: REINFORCEMENT LEARNING\n")
    print(">> EXPLAIN IDEA: CREATE MODEL A TO PREDICT HYPERPARAMETER X WHICH X IS AN OPTIMAL HYPERPARAMETER FOR MODEL B\n")
    print(">> START BY RANDOMLY PICK M, THEN REWARD THE MODEL IF M IS GOOD AND PUNISH IT IF M IS BAD.\n")
    print(">> REFERENCE: https://doi.org/10.48550/arXiv.1611.01578 (ONLY THE CONCEPT! I DIDN'T USE RNN LIKE IN THE PAPER, I USE SGD)\n")

    print("=====================================\n")

    optimal_M, optimal_f1 = fit_sgd(train_inputs, train_noisy_targets, unseen_inputs, true_targets_for_unseen_set)
    
    print(f"OPTIMAL M FOUND = {optimal_M} WITH F1-SCORE = {optimal_f1}\n")
    print("=====================================\n")

if __name__ == "__main__":
    main()
