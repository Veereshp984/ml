import pandas as pd
import math
import numpy as np

# Load data
data = pd.read_csv("id3.csv")

# Extract features
features = [feat for feat in data]
features.remove("answer")

# Node class
class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""

# Function to compute entropy
def entropy(examples):
    pos, neg = 0.0, 0.0
    for _, row in examples.iterrows():
        if row["answer"] == "yes":
            pos += 1
        else:
            neg += 1
    if pos == 0.0 or neg == 0.0:
        return 0.0
    p, n = pos / (pos + neg), neg / (pos + neg)
    return -(p * math.log(p, 2) + n * math.log(n, 2))

# Function to compute information gain
def info_gain(examples, attr):
    uniq = np.unique(examples[attr])
    gain = entropy(examples)
    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_e = entropy(subdata)
        gain -= (len(subdata) / len(examples)) * sub_e
    return gain

# Function to build the decision tree (ID3 algorithm)
def ID3(examples, attrs):
    root = Node()
    
    # Base case: If all examples have the same answer, create a leaf node
    if len(np.unique(examples["answer"])) == 1:
        root.isLeaf = True
        root.pred = examples["answer"].iloc[0]
        return root
    
    # Base case: If no attributes are left, create a leaf node with majority class
    if not attrs:
        root.isLeaf = True
        root.pred = examples["answer"].mode()[0]
        return root
    
    # Find the attribute with the maximum information gain
    max_gain, max_feat = 0, ""
    for feature in attrs:
        gain = info_gain(examples, feature)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    
    root.value = max_feat
    uniq = np.unique(examples[max_feat])
    for u in uniq:
        subdata = examples[examples[max_feat] == u]
        if entropy(subdata) == 0.0:
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = subdata["answer"].iloc[0]
            root.children.append(newNode)
        else:
            new_attrs = attrs.copy()
            new_attrs.remove(max_feat)
            child = ID3(subdata, new_attrs)
            dummyNode = Node()
            dummyNode.value = u
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    
    return root

# Function to print the decision tree
def printTree(root: Node, depth=0):
    print("\t" * depth + root.value, end="")
    if root.isLeaf:
        print(" -> " + root.pred)
    else:
        print()
    for child in root.children:
        printTree(child, depth + 1)

# Function to classify a new example
def classify(root: Node, new):
    if root.isLeaf:
        return root.pred
    for child in root.children:
        if child.value == new[root.value]:
            return classify(child.children[0], new)
    return None

# Build and display the decision tree
root = ID3(data, features)
print("Decision Tree is:")
printTree(root)

# Test with a new example
new = {"outlook": "sunny", "temperature": "hot", "humidity": "normal", "wind": "strong"}
predicted_label = classify(root, new)
print(f"Predicted label for the new example {new} is: {predicted_label}")
