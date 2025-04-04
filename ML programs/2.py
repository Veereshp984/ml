import numpy as np 
import pandas as pd

# Load the dataset
data = pd.read_csv('enjoysport.csv')
concepts = np.array(data.iloc[:, 0:-1])
target = np.array(data.iloc[:, -1])

print("\nInstances are:\n", concepts)
print("\nTarget Values are: ", target)

def learn(concepts, target): 
    # Initialize specific and general hypotheses
    specific_h = concepts[0].copy()
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    
    print("\nInitialization of specific_h and general_h")
    print("\nSpecific Boundary: ", specific_h)
    print("\nGeneral Boundary: ", general_h)
    
    for i, h in enumerate(concepts):
        print("\nInstance", i + 1, "is ", h)
        if target[i] == "yes":
            print("Instance is Positive")
            for x in range(len(specific_h)): 
                if h[x] != specific_h[x]:                    
                    specific_h[x] = '?'                     
                    general_h[x][x] = '?'            
        if target[i] == "no":            
            print("Instance is Negative")
            for x in range(len(specific_h)): 
                if h[x] != specific_h[x]:                    
                    general_h[x][x] = specific_h[x]                
                else:                    
                    general_h[x][x] = '?'        
        
        print("Specific Boundary after ", i + 1, "Instance is ", specific_h)         
        print("General Boundary after ", i + 1, "Instance is ", general_h)
        print("\n")
    
    # Remove overly general hypotheses
    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]    
    for i in indices:   
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    
    return specific_h, general_h 

# Run the learning algorithm
s_final, g_final = learn(concepts, target)

print("Final Specific_h: ", s_final, sep="\n")
print("Final General_h: ", g_final, sep="\n")