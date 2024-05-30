import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold

def train_test_svm(knl, C_values, trainX, trainY, testX, testY):
    best_C = None
    best_accuracy = 0
    kf = KFold(n_splits=5)

    for C in C_values:
        accuracies = []
        for train_index, val_index in kf.split(trainX):
            trainX_fold, trainX_val = trainX[train_index], trainX[val_index]
            trainY_fold, trainY_val = trainY[train_index], trainY[val_index]
            
            model = SVC(C=C, kernel=knl)
            model.fit(trainX_fold, trainY_fold)
            predY = model.predict(trainX_val)
            accuracies.append(accuracy_score(trainY_val, predY))

        avg_accuracy = np.mean(accuracies)
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_C = C
    
    # Train the model on the entire training set with the best C value
    final_model = SVC(C=best_C, kernel=knl)
    final_model.fit(trainX, trainY)
    testY_pred = final_model.predict(testX)
    test_accuracy = accuracy_score(testY, testY_pred)

    return {'best_C': best_C, 'best_accuracy': best_accuracy, 'test_accuracy': test_accuracy}

if __name__ == '__main__':
    # load data
    all_X, all_Y = load_svmlight_file("hw5_datasets/sonar/sonar_scale.txt")

    # load indices
    train_indices = -1 + np.loadtxt("hw5_datasets/sonar/sonar-scale-train-indices.txt", dtype=int)
    test_indices = -1 + np.loadtxt("hw5_datasets/sonar/sonar-scale-test-indices.txt", dtype=int)

    # split data
    trainX, trainY = all_X[train_indices], all_Y[train_indices]
    testX, testY = all_X[test_indices], all_Y[test_indices]
        
    # given C_values and kernels
    C_values = [0.01, 0.1, 1, 10, 100, 1000]
    kernels = ['linear', 'poly', 'rbf']

    for kernel in kernels:
        results = train_test_svm(kernel, C_values, trainX, trainY, testX, testY)
        print(f"Kernel: {kernel}")
        print(f"Best C: {results['best_C']}")
        print(f"Best Cross-Validation Accuracy: {results['best_accuracy']:.3f}")
        print(f"Test Set Accuracy: {results['test_accuracy']:.3f}")
        print()