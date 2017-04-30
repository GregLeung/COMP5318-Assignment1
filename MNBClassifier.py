import csv
import time
import math

alpha = 0.01


# Read the files
def readCsvFile(filename):
    rows = csv.reader(open(filename, "r"), delimiter=',')
    return list(rows)


# Read training CSV files with app name and tf-idf values
def readNameTfIdf(filename):
    rows = csv.reader(open(filename, "r"), delimiter=',')
    names = []
    data = list(rows)
    for i in range(len(data)):
        names.append(data[i][0])
        del data[i][0]
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


# Calculate log probability for each word by class
def wordsProb(groups):
    result = {}
    for key, value in groups.items():
        result[key] = []
        divisor = sumAll(value)
        dividend = sumWord(value)
        for d in dividend:
            quotient = d / divisor
            result[key].append(math.log(quotient, math.e))
    return result


# Calculate class prob
def labelsProb(groups):
    training_data_size = 0
    for key, value in groups.items():
        training_data_size += len(key)
    result = {}
    for key, value in groups.items():
        result[key] = len(key) / training_data_size
    return result


# Calculate probability of each class and make prediction given the test dataset and training dataset:
def predict(test_name, test_data, wordsP, labelsP):
    result = []
    for i in range(0, len(test_data)):
        maximum = float("-inf")
        label = None
        for key, values in labelsP.items():
            prob = 1.
            for j in range(len(test_data[i])):
                prob = prob * wordsP[key][j] * test_data[i][j]
            if prob > maximum:
                maximum = prob
                label = key
        print(test_name[i], label)


def main():
    # Training:
    start = time.time()
    training_names, training_data = readNameTfIdf('training_data.csv')
    labels = readCsvFile('training_labels.csv')
    end = time.time()
    print('Finish loading, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    appendClass(training_names, training_data, labels)
    end = time.time()
    print('Finish appending class, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    groups = groupByClass(training_data)
    end = time.time()
    print('Finish grouping by class, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    wordsP = wordsProb(groups)
    end = time.time()
    print('Finish calculating wordsProb, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    labelsP = labelsProb(groups)
    end = time.time()
    print('Finish calculating labelsProb, the time used was: {0} seconds'.format(end - start))

    # Make prediction:
    start = time.time()
    testing_names, testing_data = readNameTfIdf('test_data.csv')
    end = time.time()
    print('Finish loading, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    predict(testing_names, testing_data, wordsP, labelsP)
    end = time.time()
    print('Finish loading, the time used was: {0} seconds'.format(end - start))

main()
