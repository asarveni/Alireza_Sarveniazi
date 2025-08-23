import numpy as np

def tropical_distance(a, b):
    '''Berechnet die tropische Distanz zwischen zwei Vektoren'''
    return np.max(np.abs(a - b))

# Rest der Datei hier...
