import csv
import time
import math
from random import randint


# Read the selected features from preprocessing
def readSelected(filename):
    with open('Selected_features.txt') as f:
        content = [int(x.strip('\n')) for x in f.readlines()]
    return content


# Read the files
def readCsvFile(filename):
    rows = csv.reader(open(filename, "r"), delimiter=',')
    return list(rows)


# Read training CSV files with app name and tf-idf values
def readNameTfIdf(filename, alpha, selected):
    rows = csv.reader(open(filename, "r"), delimiter=',')
    names = []
    data = list(rows)
    for i in range(len(data)):
        removed = 0
        names.append(data[i][0])
        del data[i][0]
        for ele in selected:
            del data[i][ele - removed]
            removed += 1
        data[i] = [float(x) + alpha for x in data[i]]
    return names, data


# Match the name of app in the training_labels file with the name of app in the training_data file, append the
# class label at the end
def appendClass(training_name, training_data, training_label):
    for l in training_label:
        for i in range(len(training_name)):
            if l[0] == training_name[i]:
                training_data[i].append(l[1])
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


# Length normalization
def lengthNormalization(training_data):
    divisor = sumAll(training_data)
    for i in range(0, len(training_data)):
        training_data[i] = [x / divisor for x in training_data[i]]


# Calculate summation tf_idf for all words in a class
def sumAll(classtfidf):
    result = 0.
    for row in classtfidf:
        result += sum(row)
    return result


# Calculate summation of tf_idf for one word
def sumWord(classtfidf):
    result = []
    for words in zip(*classtfidf):
        result.append(sum(words))
    return result


# Calculate probability for each word by class by complement. aka. Step 4 complement
def wordsProb(groups):
    result = {}
    for targetClass, targetValue in groups.items():
        divisor = 0.
        dividend = []
        for otherClass, otherValue in groups.items():
            if targetClass != otherClass:
                divisor += sumAll(otherValue)
                dividend.append(sumWord(otherValue))
        dividend = sumWord(dividend)
        for i in range(0, len(dividend)):
            dividend[i] = math.log(dividend[i] / divisor)
        result[targetClass] = dividend
    return result


# Normalising the weight optained from step 5
def weightNormalization(wordsProb):
    for key, value in wordsProb.items():
        sum = 0.
        for ele in value:
            sum += ele
        for i in range(0, len(value)):
            value[i] = value[i] / sum


# Calculate class prob
def labelsProb(groups):
    training_data_size = 0
    for key, value in groups.items():
        training_data_size += len(key)
    result = {}
    for key, value in groups.items():
        result[key] = float(len(key)) / float(training_data_size)
    return result


# Calculate probability of each class and make prediction given the test dataset and training dataset. aka. step8
def predict(test_data, wordsP, labelsP):
    results = []
    temp = {}
    for i in range(0, len(test_data)):
        minimum = float("inf")
        label = None
        result = []
        for key, value in wordsP.items():
            prob = 0.
            for j in range(0, len(test_data[i])):
                prob += test_data[i][j] * wordsP[key][j]
            result.append(prob)
            if prob < minimum:
                minimum = prob
                label = key
        if label not in temp:
            temp[label] = 0
        temp[label] += 1
        results.append(label)
    for key, value in temp.items():
        print(key, value)
    return results


# Give each document a fold id
def giveId(training_set, foldNum):
    result = []
    temp = [len(training_set) / foldNum] * foldNum
    reminder = len(training_set) % foldNum
    temp[0] += reminder
    for i in range(0, len(training_set)):
        fold = randint(0, 9)
        while temp[fold] == 0:
            fold = randint(0, 9)
        temp[fold] -= 1
        result.append(fold)
    return result


# Pick those doc which has certain fold id from the training set
def pickFold(training_set, foldNum, foldID):
    fold = []
    foldLabel = []
    training_copy = list(training_set)
    removed = 0
    for i in range(0, len(training_set)):
        if foldID[i] == foldNum:
            foldLabel.append(training_set[i][-1])
            del training_copy[i - removed][-1]
            fold.append(training_set[i])
            del training_copy[i - removed]
            removed += 1
    return fold, foldLabel, training_copy


# find out TP, TN, FP, FN
def summary(predicted, reality):
    TP = 0.
    for i in range(len(predicted)):
        print(predicted[i], reality[i])
        if predicted[i] == reality[i]:
            TP += 1
    print(TP, len(predicted))
    print('Accuracy:', TP / float(len(predicted)))


# sub-routine for training
def training(training_data):
    start = time.time()
    groups = groupByClass(training_data)
    end = time.time()
    print('Finish grouping by class, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    wordsP = wordsProb(groups)
    end = time.time()
    print('Finish calculating wordsProb, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    weightNormalization(wordsP)
    end = time.time()
    print('Finish normalisation step, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    labelsP = labelsProb(groups)
    end = time.time()
    print('Finish calculating labelsProb step, the time used was: {0} seconds'.format(end - start))
    return wordsP, labelsP


# sub-routine for evaluation
def evaulation(training_data):
    start = time.time()
    foldID = giveId(training_data, 10)
    end = time.time()
    print('Finish assigning fold id, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    fold, foldLabel, rest = pickFold(training_data, 0, foldID)
    end = time.time()
    print('Finish picking fold, the time used was: {0} seconds'.format(end - start))
    wordP, labelsP = training(rest)
    predicted = predict(fold, wordP, labelsP)
    summary(predicted, foldLabel)


def main():
    # Loading:
    start = time.time()
    features = readSelected('Selected_features.txt')
    end = time.time()
    print('Finish loading, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    training_names, training_data = readNameTfIdf('training_data(5000).csv', math.pow(10, -1), features)
    labels = readCsvFile('training_labels.csv')
    end = time.time()
    print('Finish loading, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    lengthNormalization(training_data)
    end = time.time()
    print('Finish lengthNorm, the time used was: {0} seconds'.format(end - start))
    # pre processing:
    start = time.time()
    appendClass(training_names, training_data, labels)
    end = time.time()
    print('Finish appending class, the time used was: {0} seconds'.format(end - start))
    # Evaluation:
    evaulation(training_data)
    # Make prediction:


main()
