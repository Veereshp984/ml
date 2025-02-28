import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the dataset
hd = pd.read_csv("heart.csv")
hd = hd.replace('?', np.nan)

# Print the column names to verify
print("Column names: ", hd.columns)

# Define the Bayesian Network structure using 'target' as the label for heart disease
model = BayesianNetwork([
    ('age', 'target'),
    ('sex', 'target'),
    ('cp', 'target'),
    ('target', 'restecg'),
    ('target', 'chol')
])

# Learn the CPD using Maximum Likelihood Estimators
model.fit(hd, estimator=MaximumLikelihoodEstimator)

# Perform inference with the Bayesian Network
hd_infer = VariableElimination(model)

# Query 1: Probability of heart disease given evidence = restecg
q1 = hd_infer.query(variables=['target'], evidence={'restecg': 1})
print("\n1. Probability of Heart Disease given evidence (restecg = 1):")
print(q1)

# Query 2: Probability of heart disease given evidence = cp
q2 = hd_infer.query(variables=['target'], evidence={'cp': 2})
print("\n2. Probability of Heart Disease given evidence (cp = 2):")
print(q2)
