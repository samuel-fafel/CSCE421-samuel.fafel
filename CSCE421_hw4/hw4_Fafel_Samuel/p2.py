import numpy as np
from tqdm import tqdm
trainX_path = "hw4_datasets/20newsgroups/train.data"
trainX = np.loadtxt(trainX_path, dtype=int)
print(trainX.shape)

trainY_path = "hw4_datasets/20newsgroups/train.label"
trainY = np.loadtxt(trainY_path, dtype=int)
print(trainY.shape)

testX_path = "hw4_datasets/20newsgroups/test.data"
testX = np.loadtxt(testX_path, dtype=int)
print(testX.shape)

testY_path = "hw4_datasets/20newsgroups/test.label"
testY = np.loadtxt(testY_path, dtype=int)
print(testY.shape)


def naive_bayes_classifier(trainX, trainY, testX, testY):
    '''
    Section 1: Building up the Naive Bayes Classifier based on training data
    '''
    # compute the class probabilities
    num_classes = len(np.unique(trainY))
    print("num_classes:" + str(num_classes))
    class_counts = np.bincount(trainY)[1:] # Exclude the 0 count
    print("class_counts:" + str(class_counts)) 
    class_probs = class_counts / np.sum(class_counts)
    print("class_probs:" + str(class_probs))

    # compute the word probabilities
    all_word_ids = np.unique(np.concatenate((trainX[:, 1], testX[:, 1])))
    num_words = len(all_word_ids)
    word_counts = np.zeros((num_words, num_classes))
    for i in range(trainX.shape[0]):
        doc_id, word_id, count = trainX[i]
        class_id = trainY[doc_id-1]
        word_counts[word_id-1, class_id-1] += count
        
    word_probs = (word_counts + 1) / (np.sum(word_counts, axis=0) + num_words)
    
    '''
    Section 2: Predicting the labels for testing data
    '''
    # predict the class labels for the test documents
    predictions = []
    unique_test_docs = np.unique(testX[:, 0])
    for doc_id in unique_test_docs:
        if (doc_id % 500 == 0): print(f"Working with Doc {doc_id}")
        doc_data = testX[testX[:, 0] == doc_id]
        #print("doc_data: " +str(doc_data))
        ### Fill in the code below to do predictions for the testing data. 
        ### to fectch the probability of word "word_id" in "class_id",
        ### you need to use "word_probs[word_id-1, class_id-1]" 
        
        log_probabilities = np.log(class_probs) # Start with the log prior probability for each class
        for class_id in range(num_classes):
            for document_id, word_id, word_freq in doc_data:
                log_probabilities[class_id] += word_freq * np.log(word_probs[word_id-1, class_id])
        predicted_class = np.argmax(log_probabilities)
        predictions.append(predicted_class + 1) # Adding 1 because classes are 1-indexed
    
    # Compute the accuracy
    correct_predictions = np.sum(predictions == testY)
    accuracy = correct_predictions / len(testY)
    return accuracy

accuracy = naive_bayes_classifier(trainX, trainY, testX, testY)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:0.2f})%")




