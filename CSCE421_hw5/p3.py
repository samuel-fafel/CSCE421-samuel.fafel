import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def rescale_features(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-8)  # Adding a small constant to avoid division by zero

def mean_normalize_features(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_mean = X.mean(axis=0)
    return (X - X_mean) / (X_max - X_min + 1e-8)  # Adding a small constant to avoid division by zero

def standardize_features(X):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    return (X - X_mean) / X_std

def train_test_svm(knl, C_values, trainX, trainY, testX, testY):
    best_C = None
    best_accuracy = 0
    kf = KFold(n_splits=5)

    for C in C_values:
        accuracies = []
        f1_scores = []
        auc_scores = []
        for train_index, val_index in kf.split(trainX):
            trainX_fold, trainX_val = trainX[train_index], trainX[val_index]
            trainY_fold, trainY_val = trainY[train_index], trainY[val_index]
            
            model = SVC(C=C, kernel=knl, probability=True)
            model.fit(trainX_fold, trainY_fold)
            predY = model.predict(trainX_val)
            predY_prob = model.predict_proba(trainX_val)[:, 1]  # for AUC
            
            accuracies.append(accuracy_score(trainY_val, predY))
            f1_scores.append(f1_score(trainY_val, predY))
            auc_scores.append(roc_auc_score(trainY_val, predY_prob))

        avg_accuracy = np.mean(accuracies)
        avg_f1_score = np.mean(f1_scores)
        avg_auc_score = np.mean(auc_scores)
        
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_C = C
            best_f1_score = avg_f1_score
            best_auc_score = avg_auc_score

    # Train the model on the entire training set with the best C value
    final_model = SVC(C=best_C, kernel=knl, probability=True)
    final_model.fit(trainX, trainY)
    testY_pred = final_model.predict(testX)
    testY_pred_prob = final_model.predict_proba(testX)[:, 1]  # for AUC and ROC

    test_accuracy = accuracy_score(testY, testY_pred)
    test_f1_score = f1_score(testY, testY_pred)
    test_auc_score = roc_auc_score(testY, testY_pred_prob)

    fpr, tpr, thresholds = roc_curve(testY, testY_pred_prob)

    return {
        'best_C': best_C,
        'best_accuracy': best_accuracy,
        'best_f1_score': best_f1_score,
        'best_auc_score': best_auc_score,
        'test_accuracy': test_accuracy,
        'test_f1_score': test_f1_score,
        'test_auc_score': test_auc_score,
        'fpr': fpr,  # false positive rates for ROC
        'tpr': tpr,  # true positive rates for ROC
        'thresholds': thresholds
    }

def plot_roc_curves(fpr, tpr, label):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # Add a diagonal dashed line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # load data
    data = np.loadtxt("hw5_datasets/covtype/covtype.data", delimiter=',')
    all_X = data[:, :-1]
    all_Y = data[:, -1]

    # load indices
    train_indices = -1 + np.loadtxt("hw5_datasets/covtype/covtype.train.index.txt", dtype=int)
    test_indices = -1 + np.loadtxt("hw5_datasets/covtype/covtype.test.index.txt", dtype=int)

    # Only using part of the training data to reduce training time. 
    trainX, trainY = all_X[train_indices][0:-1:1000], all_Y[train_indices][0:-1:1000]
    testX, testY = all_X[test_indices], all_Y[test_indices]

    # convert to binary labels
    trainY[trainY!=2] = -1
    trainY[trainY==2] = 1
    testY[testY!=2] = -1
    testY[testY==2] = 1

    # preprocess training data in three different ways
    trainX_rescaled = rescale_features(trainX)
    trainX_mean_normalized = mean_normalize_features(trainX)
    trainX_standardized = standardize_features(trainX)
    
    X_min = trainX.min(axis=0)   # calculate min parameter for rescaling from training data
    X_max = trainX.max(axis=0)   # calculate max parameter for rescaling from training data
    X_mean = trainX.mean(axis=0) # calculate mean parameter for mean normalization from training data
    X_std = trainX.std(axis=0)   # calculate std parameter for standardization from training data

    # apply the transformations to the test data using parameters from the training data
    testX_rescaled = (testX - X_min) / (X_max - X_min + 1e-8) # Eliminate divide-by-zero errors
    testX_mean_normalized = (testX - X_mean) / (X_max - X_min + 1e-8)
    testX_standardized = (testX - X_mean) / (X_std + 1e-8)
   
    C_values = [0.01, 0.1, 1, 10, 100, 1000]
    kernel = 'linear'
    
    print("Getting (Rescaled) Results...")
    rescaled_results = train_test_svm(kernel, C_values, trainX_rescaled, trainY, testX_rescaled, testY)
    
    print("Getting (Mean Normalized) Results...")
    mean_normalized_results = train_test_svm(kernel, C_values, trainX_mean_normalized, trainY, testX_mean_normalized, testY)
    
    print("Getting (Standardized) Results...")
    standardized_results = train_test_svm(kernel, C_values, trainX_standardized, trainY, testX_standardized, testY)

    results = {
        'rescaled': {
            'best_C': rescaled_results['best_C'],
            'best_accuracy': rescaled_results['best_accuracy'],
            'best_f1_score': rescaled_results['best_f1_score'],
            'best_auc_score': rescaled_results['best_auc_score'],
            'test_accuracy': rescaled_results['test_accuracy'],
            'test_f1_score': rescaled_results['test_f1_score'],
            'test_auc_score': rescaled_results['test_auc_score'],
            'fpr': rescaled_results['fpr'],
            'tpr': rescaled_results['tpr'],
            'thresholds' : rescaled_results['thresholds']
        },
        'mean_normalized': {
            'best_C': mean_normalized_results['best_C'],
            'best_accuracy': mean_normalized_results['best_accuracy'],
            'best_f1_score': mean_normalized_results['best_f1_score'],
            'best_auc_score': mean_normalized_results['best_auc_score'],
            'test_accuracy': mean_normalized_results['test_accuracy'],
            'test_f1_score': mean_normalized_results['test_f1_score'],
            'test_auc_score': mean_normalized_results['test_auc_score'],
            'fpr': mean_normalized_results['fpr'],
            'tpr': mean_normalized_results['tpr'],
            'thresholds' : mean_normalized_results['thresholds']
        },
        'standardized': {
            'best_C': standardized_results['best_C'],
            'best_accuracy': standardized_results['best_accuracy'],
            'best_f1_score': standardized_results['best_f1_score'],
            'best_auc_score': standardized_results['best_auc_score'],
            'test_accuracy': standardized_results['test_accuracy'],
            'test_f1_score': standardized_results['test_f1_score'],
            'test_auc_score': standardized_results['test_auc_score'],
            'fpr': standardized_results['fpr'],
            'tpr': standardized_results['tpr'],
            'thresholds' : standardized_results['thresholds']
        }
    }
    
    print("Printing Results for each Method...")
    for method, data in results.items():
        print(f"{method}:")
        for stat, val in data.items():
            print(f"\t{stat} : {val}")
        
    # Plot ROC curve for each method
    for method, data in results.items():
        label = f"{method.capitalize()} (AUC = {data['test_auc_score']:.2f})"
        plot_roc_curves(data['fpr'], data['tpr'], label)