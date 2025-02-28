import csv
import random
import math

# Function to load the dataset from a CSV file
def loadcsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        # Converting strings into numbers for processing
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

# Function to split the dataset into training and testing sets
def splitdataset(dataset, splitratio):
    # 67% training size
    trainsize = int(len(dataset) * splitratio)
    trainset = []
    copy = list(dataset)
    while len(trainset) < trainsize:
        # Generate indices for the dataset list randomly
        index = random.randrange(len(copy))
        trainset.append(copy.pop(index))
    return [trainset, copy]

# Function to separate data by class (last column)
def separatebyclass(dataset):
    separated = {}  # Dictionary to hold class-separated data
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

# Function to calculate the mean
def mean(numbers):
    return sum(numbers) / float(len(numbers))

# Function to calculate the standard deviation
def stdev(numbers):
    if len(numbers) <= 1:  # Handle cases with insufficient data
        return 0
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

# Function to summarize the dataset
def summarize(dataset):
    # Create a list of tuples (mean, stdev) for each attribute
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]  # Exclude the label (last column)
    return summaries

# Function to summarize data by class
def summarizebyclass(dataset):
    separated = separatebyclass(dataset)
    summaries = {}
    for classvalue, instances in separated.items():
        # Summarize each class separately
        summaries[classvalue] = summarize(instances)
    return summaries

# Function to calculate probability using Gaussian distribution
def calculateprobability(x, mean, stdev):
    if stdev == 0:  # Handle zero standard deviation
        return 1 if x == mean else 0
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# Function to calculate class probabilities
def calculateclassprobabilities(summaries, inputvector):
    probabilities = {}  # Holds probabilities for each class
    for classvalue, classsummaries in summaries.items():
        probabilities[classvalue] = 1
        for i in range(len(classsummaries)):
            mean, stdev = classsummaries[i]
            x = inputvector[i]
            probabilities[classvalue] *= calculateprobability(x, mean, stdev)
    return probabilities

# Function to make predictions
def predict(summaries, inputvector):
    probabilities = calculateclassprobabilities(summaries, inputvector)
    bestLabel, bestProb = None, -1
    for classvalue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classvalue
    return bestLabel

# Function to get predictions for the test set
def getpredictions(summaries, testset):
    predictions = []
    for i in range(len(testset)):
        result = predict(summaries, testset[i])
        predictions.append(result)
    return predictions

# Function to calculate accuracy
def getaccuracy(testset, predictions):
    correct = 0
    for i in range(len(testset)):
        if testset[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testset))) * 100.0

# Main function to execute the Naive Bayes classification
def main():
    filename = 'naivedata.csv'
    splitratio = 0.67
    random.seed(42)  # Set random seed for reproducibility

    dataset = loadcsv(filename)
    trainingset, testset = splitdataset(dataset, splitratio)
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingset), len(testset)))

    # Prepare model
    summaries = summarizebyclass(trainingset)

    # Test model
    predictions = getpredictions(summaries, testset)
    accuracy = getaccuracy(testset, predictions)
    print('Accuracy of the classifier is: {0}%'.format(accuracy))

main()
