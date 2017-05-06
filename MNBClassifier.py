import csv
import time
import math
from random import randint


# Read the selected features from preprocessing
def readSelected(filePath):
    with open(filePath) as f:
        content = [int(x.strip('\n')) for x in f.readlines()]
    return content, len(content)


# Read the files
def readCsvFile(filePath):
    rows = csv.reader(open(filePath, "r"), delimiter=',')
    return list(rows)


# Read training CSV files with app name and tf-idf values
def readNameTfIdf(filePath, selected):
    rows = csv.reader(open(filePath, "r"), delimiter=',')
    names = []
    data = list(rows)
    for i in range(0, len(data)):
        names.append(data[i][0])
        # Remove the app name from the row, so whats left are tf-idf weights
        del data[i][0]
        temp = []
        # Select only the attributes in the features_selected.txt file
        for ele in selected:
            temp.append(float(data[i][ele]))
        data[i] = temp
    return names, data


# Match the name of app in the training_labels file with the name of app in the training_data file, append the
# class label at the end
def appendClass(training_name, training_data, training_label):
    for l in training_label:
        for i in range(0, len(training_name)):
            if l[0] == training_name[i]:
                training_data[i].append(l[1])
                # If the name is found break the loop early to save some running time
                break


# Group the tf-idf values by class
def groupByClass(training_data):
    group = {}
    for data in training_data:
        # Create a new key for the class
        label = data[-1]
        if label not in group:
            group[label] = []
        # Delete the class label
        del data[-1]
        # Append the document into that class
        group[label].append(data)
    return group


# Calculate summation tf_idf for all words in a class
def sumAll(classtfidf):
    result = 0.
    for row in classtfidf:
        result += sum(row)
    return result


# Calculate summation of tf_idf for one word
def sumWord(classtfidf):
    result = []
    # Use the zip function to get summation of tf-idf weights word by word
    for words in zip(*classtfidf):
        result.append(sum(words) + 1)
    return result


# Calculate probability for each word by class
def wordsProb(groups, vocLength):
    result = {}
    for label, tfidfs in groups.items():
        dividend = sumWord(tfidfs)
        divisor = sumAll(tfidfs) + vocLength
        dividend = [x / divisor for x in dividend]
        result[label] = dividend
    return result


# Calculate class prob
def labelsProb(groups):
    training_data_size = 0
    # Calculate the class size
    for key, value in groups.items():
        training_data_size += len(value)
    result = {}
    # Calculate probability of class i by dividing the count by class size
    for key, value in groups.items():
        result[key] = float(len(value)) / float(training_data_size)
    return result


# Calculate probability of each class and make prediction given the test dataset and training dataset
def predict(test_data, wordsP, labelsP):
    start = time.time()
    results = []
    for i in range(0, len(test_data)):
        # Define maximum as infinity
        maximum = float("-inf")
        label = None
        result = []
        for key, value in wordsP.items():
            prob = 0.
            for j in range(0, len(test_data[i])):
                # If tf-idf weight of word i is zero, use complement of P(wi | c)
                if math.isclose(test_data[i][j], 0, rel_tol=1e-09, abs_tol=0.0):
                    prob += math.log(1 - wordsP[key][j])
                else:
                    prob += math.log(wordsP[key][j])
                    prob += math.log(test_data[i][j])
            prob += math.log(labelsP[key])
            result.append(prob)
            # Update the maximum each time
            if prob > maximum:
                maximum = prob
                label = key
        results.append(label)
    end = time.time()
    print('Time spent in predicting:', end - start)
    return results


# Give each document a fold id
def giveId(training_set, foldNum):
    result = []
    temp = [len(training_set) / foldNum] * foldNum
    reminder = len(training_set) % foldNum
    temp[0] += reminder
    # Randomly assign just enough fold id to the documents
    for i in range(0, len(training_set)):
        fold = randint(0, foldNum - 1)
        while temp[fold] == 0:
            fold = randint(0, foldNum - 1)
        temp[fold] -= 1
        result.append(fold)
    return result


# Pick those docs has certain fold id from the training set
def pickFold(training_set, foldNum, foldID):
    fold = []
    foldLabel = []
    training_copy = []
    # Make a copy of the training_set since the training_set should not be changed
    # it is used for other fold as well.
    for doc in training_set:
        training_copy.append([x for x in doc])
    removed = 0
    for i in range(0, len(training_set)):
        if foldID[i] == foldNum:
            foldLabel.append(training_set[i][-1])
            # Remove label of the document
            del training_copy[i - removed][-1]
            fold.append(training_copy[i - removed])
            # Removed the document from copied training_set, the rest of
            # copied training_set would become the training_set for this fold
            # (i.e. other 9 folds)
            del training_copy[i - removed]
            removed += 1
    return fold, foldLabel, training_copy


# find out accuracy
def summary(predicted, reality):
    TP = 0.
    for i in range(len(predicted)):
        if predicted[i] == reality[i]:
            TP += 1
    acc = TP / float(len(predicted))
    print('Accuracy:', TP / float(len(predicted)))
    return acc


# sub-routine for loading the documents
def loading():
    start = time.time()
    features, vocLength = readSelected('..\\Input\\Selected_features.txt')
    training_names, training_data = readNameTfIdf('..\\Input\\training_data(5000).csv', features)
    test_names, test_data = readNameTfIdf('..\\Input\\test_data.csv', features)
    labels = readCsvFile('..\\Input\\training_labels.csv')
    appendClass(training_names, training_data, labels)
    end = time.time()
    print('Time spent in loading:', end - start)
    return vocLength, training_data, test_names, test_data


# sub-routine for training
def training(training_data, vocLength):
    start = time.time()
    groups = groupByClass(training_data)
    wordsP = wordsProb(groups, vocLength)
    labelsP = labelsProb(groups)
    end = time.time()
    print('Time spent in training:', end - start)
    return wordsP, labelsP


# sub-routine for evaluation
def evaulation(training_data, volLength):
    foldID = giveId(training_data, 10)
    avg = 0.
    start = time.time()
    for i in range(0, 10):
        start = time.time()
        fold, foldLabel, rest = pickFold(training_data, i, foldID)
        wordP, labelsP = training(rest, volLength)
        predicted = predict(fold, wordP, labelsP)
        end = time.time()
        avg += summary(predicted, foldLabel)
        print('Time spent in fold ', i, ':', end - start)
    end = time.time()
    print('Total time spent in evaluation:', end - start)
    print('Average accuracy:', avg / 10)


def fit(training_data, test_names, test_data, vocLength):
    wordsP, labelsP = training(training_data, vocLength)
    predicted = predict(test_data, wordsP, labelsP)
    result = []
    for i in range(0, len(predicted)):
        result.append(test_names[i] + ',' + predicted[i])
    f = open('..\\Output\\predicted_labels.csv', 'w+')
    for line in result:
        f.write(line + '\n')
    f.close()


def main():
    # Loading:
    vocLength, training_data, test_names, test_data = loading()
    training_copy = []
    for doc in training_data:
        training_copy.append([x for x in doc])
    # Make prediction:
    fit(training_copy, test_names, test_data, vocLength)
    # Evaluation:
    evaulation(training_data, vocLength)

    
main()
