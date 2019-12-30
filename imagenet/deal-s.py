import pandas as pd
import numpy as np


with open('sparsity-20.log') as f:
    #lines = (line for line in f if line.startswith('0'))
    lines = (line for line in f if line[0].isdigit())
    FH = np.loadtxt(lines, delimiter=',')
print(lines)
print(FH.shape)

tmp1 = FH.reshape(50,48)
tmp2 = np.transpose(tmp1)

print(tmp2)
print(tmp2.shape)

np.savetxt('res50-sparsity20.csv', tmp2, delimiter = ',')
