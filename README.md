# Optimising-logistic-binary-regression-models
UCL COMP0197: Applied Deep Learning coursework

Implement a specific logistic model function logistic_fun, that takes input arguments, a weight vector 𝐰, a hyperparameter representing the polynomial order 𝑀, and an input vector representing 𝐷-dimensional variable 𝐱, and returns the function value representing a positive class probability 𝑦.
```
𝑦=𝜎(𝑓𝑀(𝐱;𝐰))
```
 where 𝜎(∙) is the sigmoid function and 𝑓𝑀(𝐱;𝐰) is an 𝑀-order polynomial function of the input vector 𝐱, a linear function of the weight vector 𝐱. This function should include all the possible first- and higher-order polynomial terms including interaction terms, for example, when 𝑀=2 and 𝐷=3: 

 ```
 𝑓𝑀(𝐱;𝐰)=𝑤0+𝑤1𝑥1+𝑤2𝑥2+𝑤3𝑥3+𝑤4𝑥12+𝑤5𝑥22+𝑤6𝑥32+𝑤7𝑥1𝑥2+𝑤8𝑥1𝑥3+𝑤9𝑥2𝑥3
```

where 𝐱=[𝑥1,𝑥2,𝑥3]T is and 𝐰=[𝑤0,⋯𝑤9]T. In general, the total number of polynomial terms is 𝑝=Σ(𝐷+𝑚−1𝑚)𝑀𝑚=0. For the purpose of the coursework, you can define the order of these 𝑝 polynomial terms, as long as it is consistent throughout all the code submitted.

Implement 2 loss functions
* Cross Entropy
* Root Mean Square

Implement a stochastic minibatch gradient descent algorithm for optimising the logistic functions, fit_logistic_sgd, which takes 𝑁 pairs of 𝐱 and target values 𝑡 as input, with additional input arguments, options to choose and minimise one of the above loss functions, learning rate and minibatch size. This function returns the optimum weight vector 𝐰̂. During training, the function should report the loss periodically throughout optimisation using printed messages (minimum 10 times including beginning and end).
