import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

all_X, all_Y = load_svmlight_file("hw4_datasets/sonar/sonar_scale.txt")

train_indices = -1 + np.loadtxt("hw4_datasets/sonar/sonar-scale-train-indices.txt", dtype=int)
test_indices = -1 + np.loadtxt("hw4_datasets/sonar/sonar-scale-test-indices.txt", dtype=int)

trainX, trainY = all_X[train_indices], all_Y[train_indices]
testX, testY = all_X[test_indices], all_Y[test_indices]

c_values = [0.01, 0.1, 1, 10, 100, 1000]
best_c = c_values[0]
best_validation_error = float('inf')
validation_errors = []
training_errors = []

# Perform 5-fold cross-validation
kf = StratifiedKFold(n_splits=5)
for c in c_values:
    fold_validation_errors = []
    fold_training_errors = []
    for train_fold_index, validation_fold_index in kf.split(trainX, trainY):
        # Split the data into training fold and validation fold
        X_train_fold, X_validation_fold = trainX[train_fold_index], trainX[validation_fold_index]
        y_train_fold, y_validation_fold = trainY[train_fold_index], trainY[validation_fold_index]

        # Train the model
        model = LogisticRegression(solver='liblinear', C=c, penalty='l2')
        model.fit(X_train_fold, y_train_fold)

        # Calculate training error (1 - accuracy)
        train_fold_pred = model.predict(X_train_fold)
        train_fold_accuracy = accuracy_score(y_train_fold, train_fold_pred)
        fold_training_errors.append(1 - train_fold_accuracy)

        # Calculate validation error (1 - accuracy)
        validation_fold_pred = model.predict(X_validation_fold)
        validation_fold_accuracy = accuracy_score(y_validation_fold, validation_fold_pred)
        fold_validation_errors.append(1 - validation_fold_accuracy)

    # Average training and validation errors across folds
    avg_validation_error = np.mean(fold_validation_errors)
    avg_training_error = np.mean(fold_training_errors)

    # Save the errors for reporting
    validation_errors.append(avg_validation_error)
    training_errors.append(avg_training_error)

    # Update the best C value if this C resulted in lower validation error
    if avg_validation_error < best_validation_error:
        best_validation_error = avg_validation_error
        best_c = c

model = LogisticRegression(solver='liblinear', C=best_c, penalty='l2') # Penalty term is the regularizer term
model.fit(trainX, trainY)
predY = model.predict(testX)
accuracy = accuracy_score(testY, predY)

print(f"Values for C:\t\t\t{c_values}")
print(f"Best C:\t\t\t\t{best_c}")
print(f"Validation Errors:\t\t{validation_errors}")
print(f"Training Errors:\t\t{training_errors}")
print(f"Final Accuracy:\t\t\t{accuracy}")



