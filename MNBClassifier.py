import csv
import time
import math


# Read the files
def readCsvFile(filename):
    rows = csv.reader(open(filename, "r"), delimiter=',')
    return list(rows)


# Read training CSV files with app name and tf-idf values
def readNameTfIdf(filename, alpha):
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


# Calculate probability for each word by class by complement. aka. Step 4 complement
def wordsProb(groups):
    result = {}
    for targetClass, targetValue in groups.items():
        divisor = 0.
        dividend = []
        for otherClass, otherValue in groups.items():
            if otherClass != targetClass:
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
        result[key] = math.log(float(len(key)) / float(training_data_size))
    return result


# Calculate probability of each class and make prediction given the test dataset and training dataset. aka. step8
def predict(test_name, test_data, wordsP):
    results = []
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
        print(result)
        print(test_name[i], label)
        results.append(label)


def main():
    # Training:
    start = time.time()
    training_names, training_data = readNameTfIdf('training_data - Copy.csv', 1)
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
    weightNormalization(wordsP)
    end = time.time()
    print('Finish normalisation step, the time used was: {0} seconds'.format(end - start))

    # Make prediction:
    start = time.time()
    testing_names, testing_data = readNameTfIdf('test_data.csv', 0)
    end = time.time()
    print('Finish loading, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    predict(testing_names, testing_data, wordsP)
    end = time.time()
    print('Finish prediction, the time used was: {0} seconds'.format(end - start))


main()
