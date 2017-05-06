import csv
import time
import math


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
        data[i] = [float(x) for x in data[i]]
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


# Calculate probability of class i, given that doc contains word w
def getWordProb(groups):
    result = {}
    for label, tfIdfs in groups.items():
        result[label] = [0.] * len(tfIdfs[0])
        for doc in tfIdfs:
            for i in range(0, len(doc)):
                if doc[i] > 0:
                    result[label][i] += 1
        result[label] = [x / float(len(tfIdfs[0])) for x in result[label]]
    return result


# Calculate class prob
def labelsProb(groups):
    training_data_size = 0
    for key, value in groups.items():
        training_data_size += len(key)
    result = {}
    for key, value in groups.items():
        result[key] = float(len(key)) / float(training_data_size)
    return result


# Get fractions of docs containing a word
def getFractions(training_set):
    result = [0.] * len(training_set[0])
    for doc in training_set:
        for i in range(0, len(doc)):
            if doc[i] > 0:
                result[i] += 1.
    for i in range(0, len(result)):
        result[i] = result[i] / float(len(training_set))
    return result


# Features selection by information gain
def infoGain(classProb, wordProb, docFractions):
    result = []
    for i in range(0, len(docFractions)):
        subValue1 = 0.
        for key, value in classProb.items():
            subValue1 += value * math.log(value)
        subValue1 *= -1
        subValue2 = 0.
        subValue3 = 0.
        for key, value in wordProb.items():
            if math.isclose(wordProb[key][i], 0, rel_tol=1e-09, abs_tol=0.0):
                continue
            subValue2 += wordProb[key][i] * math.log(wordProb[key][i])
            subValue3 += (1 - wordProb[key][i]) * math.log(1 - wordProb[key][i])
        subValue2 *= docFractions[i]
        subValue3 *= (1 - docFractions[i])
        summation = subValue1 + subValue2 + subValue3
        result.append(summation)
    avg = sum(result) / len(result)
    textfile = open('Selected_features.txt', 'w+')
    for i in range(0, len(result)):
        if result[i] >= avg:
            textfile.write(str(i) + '\n')
    textfile.close()


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
    wordsP = getWordProb(groups)
    end = time.time()
    print('Finish calculating wordsProb, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    fractions = getFractions(training_data)
    end = time.time()
    print('Finish fractions calculation step, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    classProb = labelsProb(groups)
    end = time.time()
    print('Finish class probability calculation step, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    infoGain(classProb, wordsP, fractions)
    end = time.time()
    print('Finish info gain calculation step, the time used was: {0} seconds'.format(end - start))


main()
