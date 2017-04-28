import csv
import time
import math


# Read the csv files
def readCsvFile(filename):
    rows = csv.reader(open(filename, "r"), delimiter=',')
    return list(rows)


# Match the name of app in the training_labels file with the name of app in the training_data file, append the
# class label at the end, remove the name of the app afterward
def appendClass(data, label):
    for l in label:
        for d in data:
            if l[0] == d[0]:
                d.append(l[1])
                del d[0]
                break


# Convert item in the list to float
def convertToFloat(map):
    for key, value in map.items():
        for row in value:
            row = [float(x) for x in row]


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


# Calculating Mean
def meanCalculation(numbers):
    return sum(numbers) / float((len(numbers)))


# Calculating Standard deviation
def sdCaculation(numbers):
    mean = meanCalculation(numbers)
    variance = sum([pow(x - mean, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


# Calculate both mean and standard deviation of each attribute for a class, return a list of tuples(mean, std) for each
# attribute
def msCalculation(outerList):
    results = [(meanCalculation(vertical), sdCaculation(vertical)) for vertical in zip(*outerList)]
    return results


# Use msCalculation() function to find out both mean and standard deviation of each attribute for every classes
# return a new map
def msCalculationByClasses(groups):
    classMS = {}
    for key, value in groups.items():
        classMS[key] = msCalculation(value)

    return classMS


def main():
    start = time.time()
    training_data = readCsvFile('training_data - Copy.csv')
    training_labels = readCsvFile('training_labels.csv')
    # test_data = readCsvFile('test_data.csv')
    end = time.time()
    print('Finish loading, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    appendClass(training_data, training_labels)
    end = time.time()
    print('Finish appending class, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    groups = groupByClass(training_data)
    end = time.time()
    print('Finish grouping by class, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    convertToFloat(groups)
    end = time.time()
    for key, value in groups.items():
        for row in value:
            for ele in row:
                if isinstance(ele, str):
                    print('string found')
    print('Finish converting to float, the time used was: {0} seconds'.format(end - start))
    start = time.time()
    #classMS = msCalculationByClasses(groups)
    end = time.time()
    print('Finish calculating to mean and sd, the time used was: {0} seconds'.format(end - start))


main()
