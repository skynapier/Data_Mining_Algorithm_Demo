# [Quetion Introduction File](/project2.pdf)

# How to use my program?
In this project. i use Python under environment 3.6 with packages SK-learn, tensorflow-keras, cv2, numpy, DEAP.

uncompress at first.

All the data are under source root, the programs of each part are save by different dir with the name 'part1', 'part2'... 'part7'.

Some program will invoke data by the relative path dir, so that make sure data set are under the source root.

### Project file structure 

- root
- |______
-           |datasets
-           |digits
-           |jaffe
-           |part1
-           |...
-           |part7

If you want to run the program that you just need find each part program and click run.


## Part1  XOR Problem

Training set and Test set are same:
Four vectors: [1,1], [1,0], [0,1], [0,0] with four labels [0], [1], [1], [0]

Multilayer feed forward Neural network: 
Using SK-Learn MLPClassifier to solve this problem and SK-Learn accuracy_score for count the accuracy.
### Procedures:
1.	 Architecture of NN: 3-2-1. 
Learning parameters: learning rate = 0.3, No momentum using here
Activation function: logistic

2. 	Training process: 
Because the data is numeric, so that we could output class labels with the value (0,1) exactly by logistic function which could help get the accuracy of the system.
Stop criteria: convergence or reach maximum iteration (10000) times 

3.Run many times to find the random seed that could always solve XOR problem

GP:
EC technique do not assume we have domain knowledge, however, we have it then we need to reduce the search space. Regarding XOR problem as a logistic 
problem for reduce the search space, which support by all of the values in the vectors and labels are 0 or 1 and could convert into True and False. 

Using DEAP for GP and same random seed with NN above.
Procedures:
1.	Set it as maximizing problem which means try best to find programs that fit all vectors
2.	Terminal Set, X1 and X2
Function Set: {and, or and not}.
Fitness function: number of correct classified vectors, so that the maximum fitting case is fitness function return 4.
3.	Crossover rate = 0.8, Mutation rate = 0.1, and reproduction rate = 0.1
4.	Population size = 30, number of generations = 30
5.	Stopping criteria: reach to the 30 times of iteration

## Part2  Digit Recognition

I Using SK-learn MLPClassifier to solve this problem. I do the task digit00, digit15, digit30, digit50 and digit60. 

### Procedures:
1.	Load data and preprocessing the data:
Using DataFrame in Pandas as data structure and split 50% as training set and remainders as test set.

2.	Classifier:
I try many times (> 100) to find the best parameters for each classifier.
KNN with number of neighbours = 7.
MLP with architecture 49 - 30 - 1. Learning rate = 0.04, activation function = relu , stopping until convergence.

3.	Metric
I choose F1-score as performance metric, because I think that in this problem F1-score could show the reflect directly rather than AUC. In this case the class = 10, 
so that just looking on TPR/FPR is not good enough. F1-score is weight by the number of instances of each class.

4.	The purpose:
I want to find is it the accuracy will depend on the noisy ratio and is that KNN is worth than MLP.

## Part3 Image Classification using Neural Networks

Multilayer feed forward Neural network: 
I use the SK-Learn OpenCV, ShuffleSplit, MLPand train_test_split methods to support my thinking.
### Procedures:
1.	Load data and preprocessing the data:
I random select 70% of data as training set and remainders as test set by train_test_split. I do not want to using row pixels so that I select 8 features by what I did in project1 
and through statistic get each feature¡¯s mean and stand deviation. So that we have vectors contain 16 values and 1 class label.

2.	Classifier
I use K-Fold algorithm to avoid overfitting for MLP with architecture 17-8-7-1. Solver = SGD, momentum = 0.3, learning rate= 0.04, activation =logistic

I run 10000 times to find best parameters for MLP which could avoid overfitting. I find that the learning is more and more relay on training set, so that I found that using random 
seed = 4555 to pick training set and seed = 2834 for SGD learning, which could reach to best performance.

3.	Metric 
I use accuracy_score in SK-learn to reflect the model and plot the learning curve.

CNN:
I using tensorflow.kersa for build CNN model.

### Procedures:
1.	Load data and preprocessing the data:
In CNN the training and test data are different with NN, it uses whole image rather than do feature statistic extraction. I random pick 70% data as training set and remainders as test set.

2.	Architecture:
First layer is 32 filters with kernel size = (3,3) and activation function = relu 
Second layer is 64 filters with kernel size = (3,3) and activation function = relu
Third layer is Maxpooling with 2*2
Forth layer is Dropout layer with rate = 0.25
Fifth layer is Flatten layer
Sixth layer is Dense fully connected layer with activation function = relu
Seventh layer is Dropout layer with rate = 0.5
Final layer is Dense fully connected layer with activation function = softmax
Relu will be fast to compute and fix gradient (1) when positive. The first two layer is convolutions then subsampling at third layer. Forth layer and seventh layer are drop out during 
the training process at each time before updating the weight for change the network structure/ 

3.	Optimizer
The Loss function using categorical_crossentropy, using mini batch optimizer with batch size = 9 and epochs = 18 learning rate = 1.0, and using validation set to avoid overfitting. 
Stop until model convergence. 

## Part4 Symbolic regression problem 

I use DEAP for solve symbolic regression problem. 

Although we know that the range of this function is [-inf, inf ], However GP cannot using this as search space, because, it is quite big. In this problem, I set 1000 numbers belongs 
to [-10, 10] as its search space.


## Part5 Function Optimisation (PSO)

Find the minimum of Rosenbrock's function and Griewanks's function:

I use DEAP for solve optimisation problem. 
Procedures:
1.	Set it as minimize problem which means search 
2.	Both C1, C2 are set dynamic from 0 to 1.5. W is also dynamic from -1 to 1 and the population size = 1000.
3.	The topology in this program is star which means gbest, each particle is influenced by the best found from the entire swam
4.	Stopping criteria is generation until 200 times.

## Part6 Feature Construction 
I use GP, SK-Learn in this part. The features in both balance.data and wine.data are numeric type. What I have done in this part is using GP for combine multiple features 
into one feature and then use Naive Bayes and Decision Tree classify algorithms to solve classification which could using cross validation to find the proper accuracy.

Procedures:
1.	Terminal set is {+, -, *, number in one decimal range from -1 to 1}. 
2.	Fitness function which I using predict accuracy, however, I do not want to use NB or DT to achieve it, which might cause over-fitting, and I use KNN instead. 
KNN need a small time for a classification problem.
Crossover rate = 0.8, Mutation rate = 0.15, and reproduction rate = 0.05
Limit the maximum of the tree height = 15
Tournament selection with size = 3
3.	Population size = 300
4.	Stop criteria: 
For the balance data: stop until iterate 10 times
For the wine data: stop until iterate 15 times.

## Part7 Feature Selection (Information Gain and Chi-square)

I use SK-learn in this part instead of WEKA. Mutal_info_classif method for Information Gain algorithm, chi2 for Chi-Square algorithm and RFE for wrapper algorithm.

Procedures:
1.	WBC.data hold 30 features and Sonar.data hold 60 features
2.	Na?ve Bayes algorithm for classification 
3.	I select top 5 features for Information Gain, Chi-square and Wrapper algorithms by order these in descending order.
4.	I use both accuracy and F1-score for metric.




