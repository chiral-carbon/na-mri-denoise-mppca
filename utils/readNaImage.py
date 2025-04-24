import numpy as np

crID = {'real': 1, 'complex': 2}

def read_na_image(fName, szNX, szNY, szNZ, dtype='complex'):
    with open(fName, 'rb') as file:
        data = np.fromfile(file, dtype=np.float32, count=szNX * szNY * szNZ * crID['complex'])

    rEcho = data[::2]  # Real parts at even positions
    iEcho = data[1::2]  # Complex parts at odd positions
    
    cEcho = rEcho + 1j * iEcho
    
    imaData = cEcho.reshape((szNX, szNY, szNZ))
    
    return imaData