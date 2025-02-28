import csv

# Initialize an empty list to hold the dataset
a = []

# Read the CSV file
with open('sports.csv', 'r') as csvfile:
    next(csvfile)  # Skip the header row
    for row in csv.reader(csvfile):
        a.append(row)
    print(a)

# Print the number of training instances
print("\nThe total number of training instances are: ", len(a))

# Determine the number of attributes
num_attribute = len(a[0]) - 1

# Initialize the hypothesis
print("\nThe initial hypothesis is: ")
hypothesis = ['0'] * num_attribute
print(hypothesis)

# Iterate through the instances in the dataset
for i in range(0, len(a)):
    if a[i][num_attribute] == 'yes':
        print("\nInstance ", i + 1, "is", a[i], " and is a Positive Instance")
        for j in range(0, num_attribute):
            if hypothesis[j] == '0' or hypothesis[j] == a[i][j]:
                hypothesis[j] = a[i][j]
            else:
                hypothesis[j] = '?'
        print("The hypothesis for the training instance", i + 1, " is: ", hypothesis, "\n")
    elif a[i][num_attribute] == 'no':  # Use 'elif' to avoid redundant checks
        print("\nInstance ", i + 1, "is", a[i], " and is a Negative Instance Hence Ignored")
        print("The hypothesis for the training instance", i + 1, " is: ", hypothesis, "\n")

# Print the final maximally specific hypothesis
print("\nThe Maximally specific hypothesis for the training instance is: ", hypothesis)
