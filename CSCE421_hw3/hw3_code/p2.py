from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import math
import time

def load_data(trainX_path, trainY_path, testX_path, testY_path):
    # THE FOLLOWING CODE WAS PROVIDED (slight edits made)
    
    trainX = np.loadtxt(trainX_path, dtype=float)
    train_mean = np.mean(trainX, 0)
    trainX = trainX - train_mean

    # the returned eigenvalue by np.linalg.eigh is sorted in ascending order
    eigenvalue, eigenvector = np.linalg.eigh(trainX.T.dot(trainX))

    trainY = np.loadtxt(trainY_path, dtype=float)
    
    testX = np.loadtxt(testX_path, dtype=float)
    testX = testX - train_mean

    testY = np.loadtxt(testY_path, dtype=float)
    
    return trainX, trainY, testX, testY, eigenvalue, eigenvector
    
def cross_validation(trainX, trainY, folds):
    """Problem 1 - Part 1:
        Train a kNN based on the original features. You should conduct cross-validation (of your choice) to select the best k.
    """
    # Define cross-validation strategy
    cv_method = KFold(n_splits=folds, random_state=None, shuffle=True) # Define the split into 5 folds

    # Initialize list to store average cross-validation scores for different values of k
    cv_scores = []

    # Perform cross cross-validation for range of k values
    for k in range(1, 20):
        # Initialize kNN classifier with the current k
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Perform cross-validation (maximize accuracy = minimize error)
        scores = cross_val_score(knn, trainX, trainY, cv=cv_method, scoring='accuracy')
        
        # Compute the average accuracy and store it
        cv_scores.append((k, scores.mean()))

    # Find the k value with the highest average accuracy (lowest average error)
    best_k = max(cv_scores, key=lambda x: x[1])[0]

    # Output the best k value and its accuracy
    accuracy = cv_scores[best_k-1][1]
    percent = accuracy * 100
    print("\tBest k =", best_k)
    print(f"\tAccuracy: {percent:0.2f}% ({cv_scores[best_k-1]})")
    return best_k, accuracy

def PCA_cross_validation(trainX, trainY, testX, testY, folds):
    d_values = range(1, 10) # Range of d values to test
    k_values = range(1, 10) # Range of k values for kNN to test
    
    # Define cross-validation strategy
    cv_method = KFold(n_splits=folds, random_state=None, shuffle=True) # Define the split into 5 folds

    best_d = None
    best_k = None
    best_score = 0

    # Perform cross-validation
    for d in d_values:
        # Apply PCA with d components
        pca = PCA(n_components=d)
        pca.fit(trainX)
        trainX_pca = pca.transform(trainX)

        for k in k_values:
            # Initialize kNN classifier with the current k
            knn = KNeighborsClassifier(n_neighbors=k)
            
            # Perform cross-validation (maximize accuracy = minimize error)
            scores = cross_val_score(knn, trainX_pca, trainY, cv=cv_method, scoring='accuracy')

            # Check if the current score is better than the best score
            average_score = np.mean(scores)
            if average_score > best_score:
                best_d = d
                best_k = k
                best_score = average_score

    # Train final model with best_d and best_k on full training set
    pca = PCA(n_components=best_d)
    pca.fit(trainX)
    trainX_pca = pca.transform(trainX)
    X_test_pca = pca.transform(testX)

    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(trainX_pca, trainY)
    print("\tbest k:", best_k)
    print("\tbest d:", best_d)
    print("\tbest score:", best_score)
    return best_k, best_d, best_score

def plot_data(trainX, trainY, testX, testY, eigenvalue, eigenvector, cv_k, cv_d):
    # THE FOLLOWING CODE WAS PROVIDED (slight edits made)
    
    d = cv_d # You need to do cross validation to choose the best d    
    # the last d eigen vectors are corresponding to the 
    # d largest eigen values
    pca_trainX = trainX.dot(eigenvector[:,-d:])
    pca_testX = testX.dot(eigenvector[:,-d:])

    # Project to the direction correponding to the largest eigenvalue
    plt.figure(1) 
    plt.plot(pca_trainX[trainY==1, -1], np.zeros(pca_trainX[trainY==1, -1].shape[0]), 'x', color='r')
    plt.plot(pca_trainX[trainY==-1, -1], np.zeros(pca_trainX[trainY==-1, -1].shape[0]), 'o', color='b')
    plt.xlabel("d1")

    # Project to the two directions correponding to the two largest eigenvalues
    plt.figure(2)
    plt.plot(pca_trainX[trainY==1, -1], pca_trainX[trainY==1, -2], 'x', color='r')
    plt.plot(pca_trainX[trainY==-1, -1], pca_trainX[trainY==-1, -2], 'o', color='b')
    plt.xlabel("d1")
    plt.ylabel("d2")

    # Project to the three directions correponding to the three largest eigenvalues
    fig = plt.figure(3)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pca_trainX[trainY==1, -1], pca_trainX[trainY==1, -2], \
            pca_trainX[trainY==1, -3], 'x', color='r')

    ax.scatter(pca_trainX[trainY==-1, -1], pca_trainX[trainY==-1, -2], \
            pca_trainX[trainY==-1, -3], 'o', color='b')
    ax.set_xlabel('d1')
    ax.set_ylabel('d2')
    ax.set_zlabel('d3')

    accuracy = []
    k = cv_k # You need to do cross validation to choose the best k
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(pca_trainX[:, -d:], trainY)
    print("With d = 3, k=50, testing accuracy = "  \
            +str(knn.score(pca_testX, testY, sample_weight=None)) )

    plt.show()

if __name__ == '__main__':
    trainX_path = "../hw3_datasets/gisette/gisette_trainSet.txt"
    trainY_path = "../hw3_datasets/gisette/gisette_trainLabels.txt"
    testX_path = "../hw3_datasets/gisette/gisette_testSet.txt"
    testY_path = "../hw3_datasets/gisette/gisette_testLabels.txt"
    
    load_start = time.time()
    print("Loading data...", end='', flush=True)
    trainX, trainY, testX, testY, eigenvalue, eigenvector = load_data(trainX_path, trainY_path, testX_path, testY_path)
    load_end = time.time()
    print(f" Done. Took {(load_end - load_start):0.3f} seconds")
    
    folds = 5
    # Part 1
    if False:
        print(f"Performing {folds}-Fold Cross Validation...")
        cv_start = time.time()
        best_k, accuracy = cross_validation(trainX, trainY, folds)
        cv_end = time.time()
        print(f"\tTook {(cv_end-cv_start):0.3f} seconds")
    
    # Part 2
    if False:
        print(f"Performing PCA and {folds}-Fold Cross Validation...")
        PCA_start = time.time()
        k, d, accuracy_PCA = PCA_cross_validation(trainX, trainY, testX, testY, folds)
        PCA_end = time.time()
        print(f"\tTook {(PCA_end-PCA_start):0.3f} seconds")
    
    # Part 3
    print(f"Plotting with k=5, d=1")
    plot_data(trainX, trainY, testX, testY, eigenvalue, eigenvector, 5, 3)

    print(f"Total Execution Time: {(time.time()-load_start):0.2f} seconds")

