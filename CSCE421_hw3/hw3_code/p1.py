import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def load_data(trainX_path, trainY_path, testX_path):
    # THE FOLLOWING CODE WAS PROVIDED (slight edits made)
    trainX_path = trainX_path
    trainX = np.loadtxt(trainX_path, dtype=float, encoding=None, delimiter=",")
    #print("trainX " + str(trainX))
    print("\ttrainX.shape " + str(trainX.shape))

    trainY_path = trainY_path
    trainY = np.loadtxt(trainY_path, dtype=float, encoding=None, delimiter=",")
    #print("trainY " + str(trainY))
    print("\ttrainY.shape " + str(trainY.shape))

    testX_path = testX_path
    testX = np.loadtxt(testX_path, dtype=float, encoding=None, delimiter=",")
    #print("testX " + str(testX))
    print("\ttestX.shape " + str(testX.shape))
    
    return trainX, trainY, testX

def leave_one_out(trainX, trainY):
    """Problem 1 - Part 1:
        Use the leave one out cross validation on the training data to select 
        the best k among {1, 2, ..., 10}. Report the averaged leave-one-out error 
        (averaged over all training data points) for each k ∈ {1, 2, ..., 10}.
    """
    # Define leave-one-out cross-validation strategy
    cv_method = KFold(n_splits=trainX.shape[0])

    # Initialize list to store average cross-validation scores for different values of k
    cv_scores = []

    # Perform leave-one-out cross-validation for k values from 1 to 10
    for k in range(1, 11):
        # Initialize kNN classifier with the current k
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Perform leave-one-out cross-validation (maximize accuracy = minimize error)
        scores = cross_val_score(knn, trainX, trainY, cv=cv_method, scoring='accuracy')
        
        # Compute the average accuracy and store it
        cv_scores.append((k, scores.mean()))

    # Find the k value with the highest average accuracy (lowest average error)
    best_k = max(cv_scores, key=lambda x: x[1])[0]

    # Output the best k value and the leave-one-out error for all k values
    print("\tk: error\t\t(percent)")
    for cv_score in cv_scores:
        k = cv_score[0]
        error = 1-cv_score[1]
        percent = error*100
        print(f"\t{k}: {error}\t({percent:0.2f}%)") # 1-accuracy = error
    print("\tBest k =", best_k)
    return best_k, cv_scores

def predict_test_labels(trainX, trainY, testX, best_k):
    """Problem 1 - Part 2:
        Based on Part 1, use the best k to predict the class labels for test instances. 
        You should also report the predicted labels for the testSet.
    """
    # Initialize the kNN classifier with the best k value
    knn_best = KNeighborsClassifier(n_neighbors=best_k)

    # Train the classifier on the entire training set
    knn_best.fit(trainX, trainY)

    # Predict the class labels for the test set
    predicted_labels = knn_best.predict(testX)

    # Return the predicted labels
    print(f"\t{predicted_labels.tolist()}")
    return predicted_labels

if __name__ == '__main__':
    heart_trainX_path = "../hw3_datasets/heart/heart_trainSet.txt"
    heart_trainLabels_path = "../hw3_datasets/heart/heart_trainLabels.txt"
    heart_testX_path = "../hw3_datasets/heart/heart_testSet.txt"
    
    print("Loading data...")
    trainX, trainLabels, testX = load_data(heart_trainX_path, heart_trainLabels_path, heart_testX_path)
    print("Completing 'Leave-One-Out' Cross Validation...")
    best_k, cv_scores = leave_one_out(trainX, trainLabels)
    print("Predicting Test Labels...")
    predicted_labels = predict_test_labels(trainX, trainLabels, testX, best_k)
