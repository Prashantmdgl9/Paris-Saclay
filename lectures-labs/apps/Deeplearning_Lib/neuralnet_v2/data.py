import numpy as np

def data():
    # X = (hours sleeping, hours studying), y = test score of the student
    X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
    y = np.array(([92], [86], [89]), dtype=float)

    # scale units
    X = X/np.amax(X, axis=0) #maximum of X array
    y = y/100 # maximum test score is 100
    return X,y
