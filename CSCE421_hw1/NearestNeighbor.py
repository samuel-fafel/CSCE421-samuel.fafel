"""
CSCE421-501 (Machine Learning)
02/08/2024
Samuel Fafel

Answer to Question 5.2:
    What percentage of neighbor data pairs have the same labels (yi = yj)?
    
    Using 10 Data Points:
        Euclidean Distance: 100%
        Manhattan Distance: 100%
        Cosine Distance:    100%
    
    Using 100 Data Points:
        Euclidean Distance: 96.0%
        Manhattan Distance: 97.0%
        Cosine Distance:    96.0%
        
    Using 1,000 Data Points:
        Euclidean Distance: 94.8%
        Manhattan Distance: 93.7%
        Cosine Distance:    94.7%
        
    Using Maximum Data Points:
        Euclidean Distance: 95.0%
        Manhattan Distance: 94.1%
        Cosine Distance:    94.9%
"""
from sklearn.datasets import load_svmlight_file as lsf
import numpy as np

def load_dataset(train_filepath, test_filepath):
    # Load via sklearn
    x_train, y_train = lsf(train_filepath) # training data features, training data labels
    x_test, y_test = lsf(test_filepath) # testing data features, testing data labels
    
    # Convert to dense matrices
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    return x_train, y_train, x_test, y_test

def euclidean_distance(a, b):
    E_distance = np.sqrt(np.sum((a - b)**2))
    return E_distance

def manhattan_distance(a, b):
    M_distance = np.sum(np.abs(a - b))
    return M_distance

def cosine_distance(a, b):
    sum_ab = np.sum(a*b)
    sum_a_squared = np.sum(a**2)
    sum_b_squared = np.sum(b**2)
    C_distance = 1 - (sum_ab / (np.sqrt(sum_a_squared) * np.sqrt(sum_b_squared)))
    return C_distance

def find_nearest_neighbors(x_train, y_train, x_test, y_test, distance_function, max_samples):
    neighbor_pairs = []
    
    samples = len(y_test) if len(y_test) < max_samples else max_samples
    for i in range(samples): # For each xi in testing set, up to max_samples...
        distances = []
        for train_sample in x_train: # calculate distances to all points in training set.
            distances.append(distance_function(x_test[i], train_sample))
        
        nearest_neighbor_index = np.argmin(distances) # Find minimum distance index
        pair = [(x_test[i], y_test[i]), (x_train[nearest_neighbor_index], y_train[nearest_neighbor_index]), distances[nearest_neighbor_index]]
        neighbor_pairs.append(pair) # [ (xi,yi), (xj,yj), distance ]
    return neighbor_pairs

def calculate_accuracy(neighbor_pairs, verbose=True):
    matches = 0
    if verbose: print("\n\tNeighbor Labels:")
    for i in range(len(neighbor_pairs)):
        pair_i = neighbor_pairs[i][0] # Extract (xi,yi)
        pair_j = neighbor_pairs[i][1] # Extract (xj,yj)
        distance = neighbor_pairs[i][2] # Extract distance from i to j 
        
        if verbose: print(f"\t  Yi: {int(pair_i[1]):>2}  -  Yj: {int(pair_j[1]):>2} (distance {distance:>6.3f})", end='')
        if pair_i[1] == pair_j[1]: # Actual Yj == Predicted Yj
            matches += 1
            if verbose: print(" | MATCH!")
        else:
            if verbose: print()
    accuracy = (matches / len(neighbor_pairs)) * 100
    return accuracy

def main():
    train = 'data/usps' # Training Data
    test = 'data/usps.t' # Testing Data
    print("Loading Training and Testing Data... ", end='', flush=True)
    x_train, y_train, x_test, y_test = load_dataset(train, test)
    print("Done")
    print("-------")
    
    max_samples = 10 # test at most X samples
    print(f"Max Samples: {max_samples}")
    for distance_function in [euclidean_distance, manhattan_distance, cosine_distance]:
        print(f"Using {distance_function.__name__}: ", end='', flush=True)
        neighbor_pairs = find_nearest_neighbors(x_train, y_train, x_test, y_test, distance_function, max_samples)
        accuracy = calculate_accuracy(neighbor_pairs, verbose=False)
        print(f"\tMatching Labels: {accuracy}%\n")
        
if __name__ == '__main__':
    main()