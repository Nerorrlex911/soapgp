import numpy as np
from scipy.sparse import csr_matrix, vstack

if __name__ == "__main__":
    # 创建一些稀疏矩阵
    sparse_matrix_1 = csr_matrix(np.array([
        [1, 0, 0],
        [0, 2, 0]
    ]))

    sparse_matrix_2 = csr_matrix(np.array([
        [0, 0, 3],
        [4, 0, 0]
    ]))

    # 将稀疏矩阵放入列表中
    sparse_matrix_list = [sparse_matrix_1, sparse_matrix_2]

    # 将列表中的稀疏矩阵垂直堆叠成一个整体的稀疏矩阵
    combined_sparse_matrix = vstack(sparse_matrix_list)

    # 打印结果
    print(combined_sparse_matrix)