import pandas as pd
import numpy as np


with open('sparsity-13.log') as f:
    lines = (line for line in f if line.startswith('0'))
    FH = np.loadtxt(lines, delimiter=',', skiprows=1)


FHF = np.insert(FH,0,0.29528144914276744,0)
tmp1 = FHF.reshape(50,51)
tmp2 = np.transpose(tmp1)

print(tmp2)
print(tmp2.shape)

np.savetxt('sparsity13.csv', tmp2, delimiter = ',')