from sklearn.datasets import load_svmlight_file as lsf
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
import numpy as np

def load_dataset(train_filepath, test_filepath):
    # Load training data via sklearn
    x_train, y_train = lsf(train_filepath)
    
    # Find out the number of features in the training data
    n_features = x_train.shape[1]
    
    # Load testing data, specifying the number of features
    x_test, y_test = lsf(test_filepath, n_features=n_features)

    return x_train, y_train, x_test, y_test

def fit_ridge_regression(x_train, y_train, x_test, y_test, lmd):
    print(f"Evaluating Ridge for Lambda = {lmd}")
    # Fit Ridge regression
    ridge_w = Ridge(alpha=lmd)
    ridge_w.fit(x_train, y_train)
    ridge_coefs = ridge_w.coef_
    
    # Count non-zero coefficients & report
    non_zero_ridge = np.sum(ridge_coefs != 0)
    print(f'\tNumber of non-zero coefficients in Ridge: {non_zero_ridge}')
    
    # return ridge model and non-zero coefficient count
    ridge_package = (ridge_w, non_zero_ridge)
    return ridge_package

def fit_lasso(x_train, y_train, x_test, y_test, lmd):
    print(f"Evaluating Lasso for Lambda = {lmd}")
    # Fit Lasso regression
    n_samples = x_train.shape[0]
    lasso_w = Lasso(alpha=lmd / n_samples, max_iter=5000) # max_iter ensures termination for small lambda
    lasso_w.fit(x_train, y_train)
    
    # Count non-zero coefficients & report
    non_zero_lasso = np.sum(lasso_w.coef_ != 0)
    print(f'\tNumber of non-zero coefficients in Lasso: {non_zero_lasso}')
    
    # return lasso model and non-zero coefficient count
    lasso_package = (lasso_w, non_zero_lasso)
    return lasso_package

def evaluate_models(x_train, y_train, x_test, y_test, lambdas):
    non_zeros_ridge = []
    ridge_train_errors = []
    ridge_test_errors = []
    
    non_zeros_lasso = []
    lasso_train_errors = []
    lasso_test_errors = []

    for lmd in lambdas:
        # Get Ridge Regression data:
        ridge_package = fit_ridge_regression(x_train, y_train, x_test, y_test, lmd)
        ridge_w = ridge_package[0]
        non_zeros_ridge.append(ridge_package[1])
        # Predict and find error
        ridge_train_pred = ridge_w.predict(x_train)
        ridge_test_pred = ridge_w.predict(x_test)
        ridge_train_errors.append(root_mean_squared_error(y_train, ridge_train_pred))
        ridge_test_errors.append(root_mean_squared_error(y_test, ridge_test_pred))
        
        # Get Lasso data:
        lasso_package = fit_lasso(x_train, y_train, x_test, y_test, lmd)
        lasso_w = lasso_package[0]
        non_zeros_lasso.append(lasso_package[1])  
        # Predict and find error
        lasso_train_pred = lasso_w.predict(x_train)
        lasso_test_pred = lasso_w.predict(x_test)     
        lasso_train_errors.append(root_mean_squared_error(y_train, lasso_train_pred))
        lasso_test_errors.append(root_mean_squared_error(y_test, lasso_test_pred))     

    # Report the best lambda based on test RMSE
    best_lambda_ridge = lambdas[np.argmin(ridge_test_errors)]
    print(f"Best lambda for Ridge: {best_lambda_ridge} with Test RMSE: {min(ridge_test_errors)}")
    best_lambda_lasso = lambdas[np.argmin(lasso_test_errors)]
    print(f"Best lambda for Lasso: {best_lambda_lasso} with Test RMSE: {min(lasso_test_errors)}")

    print("Showing Plots...")
    plt.figure(figsize=(18, 12))
    # Ridge RMSE plot
    plt.subplot(2, 2, 3)
    plt.plot(lambdas, ridge_train_errors, label='Ridge Train RMSE')
    plt.plot(lambdas, ridge_test_errors, label='Ridge Test RMSE')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('RMSE')
    plt.title('Ridge Regression RMSE vs Lambda')
    plt.legend()

    # Lasso RMSE plot
    plt.subplot(2, 2, 4)
    plt.plot(lambdas, lasso_train_errors, label='Lasso Train RMSE')
    plt.plot(lambdas, lasso_test_errors, label='Lasso Test RMSE')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('RMSE')
    plt.title('Lasso RMSE vs Lambda')
    plt.legend()

    # Plotting number of non-zero weights for Ridge
    plt.subplot(2, 2, 1)
    plt.plot(lambdas, non_zeros_ridge, marker='o', label='Ridge Non-Zero Weights')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Number of Non-Zero Weights')
    plt.title('Ridge Non-Zero Weights vs Lambda')
    plt.legend()

    # Plotting number of non-zero weights for Lasso
    plt.subplot(2, 2, 2)
    plt.plot(lambdas, non_zeros_lasso, marker='o', label='Lasso Non-Zero Weights')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Number of Non-Zero Weights')
    plt.title('Lasso Non-Zero Weights vs Lambda')
    plt.legend()

    # Display the plots
    plt.tight_layout()  # Adjusts the plots to fit into the figure area.
    plt.grid(True)
    plt.show()
    
    return

def cross_validate_and_test(x_train, y_train, x_test, y_test, lambdas):
    # Define a grid of lambda values for Ridge and Lasso
    print("Part i")
    ridge_params = {'alpha': lambdas}
    lasso_params = {'alpha': [l / x_train.shape[0] for l in lambdas]}  # Adjust for the number of samples for Lasso

    # Set up the Ridge grid search
    print("Part ii")
    ridge_search = GridSearchCV(Ridge(), ridge_params, scoring='neg_root_mean_squared_error', cv=5)
    print("\tPart iia")
    ridge_search.fit(x_train, y_train)

    # Set up the Lasso grid search
    print("Part iii")
    lasso_search = GridSearchCV(Lasso(max_iter=5000), lasso_params, scoring='neg_root_mean_squared_error', cv=5)
    print("\tPart iiia")
    lasso_search.fit(x_train, y_train)

    # Retrieve the best lambda values
    print("Part iv")
    best_lambda_ridge = ridge_search.best_params_['alpha']
    best_lambda_lasso = lasso_search.best_params_['alpha'] * x_train.shape[0]  # Scale back the lambda for Lasso

    # Retrain models on the entire training dataset using the best lambda values
    print("Part v")
    best_ridge_model = Ridge(alpha=best_lambda_ridge).fit(x_train, y_train)
    best_lasso_model = Lasso(alpha=best_lambda_lasso, max_iter=5000).fit(x_train, y_train)

    # Predict and evaluate on the testing data
    print("Part vi")
    ridge_test_pred = best_ridge_model.predict(x_test)
    lasso_test_pred = best_lasso_model.predict(x_test)

    # Compute the RMSE for both models
    print("Part vii")
    ridge_test_rmse = root_mean_squared_error(y_test, ridge_test_pred)
    lasso_test_rmse = root_mean_squared_error(y_test, lasso_test_pred)

    # Return the results
    return {
        'best_lambda_ridge': best_lambda_ridge,
        'best_lambda_lasso': best_lambda_lasso,
        'ridge_test_rmse': ridge_test_rmse,
        'lasso_test_rmse': lasso_test_rmse
    }

def main():
    #------------------------------------------------------------------
    train = 'data/E2006_train' # Training Data
    test = 'data/E2006_test' # Testing Data
    print("--------------")
    print("Loading Training and Testing Data... ", end='', flush=True)
    x_train, y_train, x_test, y_test = load_dataset(train, test)
    print("Done")
    #------------------------------------------------------------------
    print("--- PART 1 ---")
    lmd = 0.1
    #ridge_package = fit_ridge_regression(x_train, y_train, x_test, y_test, lmd)
    #lasso_package = fit_lasso(x_train, y_train, x_test, y_test, lmd)
    print("--------------")
    #------------------------------------------------------------------
    print("--- PART 2 ---")
    lambdas = [0.1, 1, 100, 1e3, 1e4] # Skipping small values of Lambda because Lasso takes too long
    evaluate_models(x_train, y_train, x_test, y_test, lambdas)
    print("--------------")
    #------------------------------------------------------------------
    print("--- PART 3 ---")
    results = cross_validate_and_test(x_train, y_train, x_test, y_test, lambdas)
    print(results)
    print("--------------")
    #------------------------------------------------------------------
    
if __name__ == '__main__':
    main()