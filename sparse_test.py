import numpy as np
from scipy.sparse import csr_matrix, vstack

if __name__ == "__main__":
    data = np.load("sigma2_soap_1.npz")
    print(data.files)
    for i,f in enumerate(data.files):
        print(f)
        print(data[f].shape)
        print(data[f])