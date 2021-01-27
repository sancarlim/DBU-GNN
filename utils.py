
import numpy as np
import torch

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_norm = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_norm

#################################################################################
#############################  GRIP  #############################################

def get_adjacency(A):
    # compute hop steps
    num_node = A.shape[1]
    max_hop=2
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d

    # compute adjacency
    valid_hop = range(0, max_hop + 1)
    adjacency = np.zeros((num_node, num_node))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    
    A= adjacency

    #       def normalize_adjacency(self, A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)

    valid_hop = range(0, max_hop + 1)
    A = np.zeros((len(valid_hop), num_node, num_node))
    for i, hop in enumerate(valid_hop):
        A[i][hop_dis == hop] = AD[hop_dis == hop]
    return A

#################################################################################
#################################################################################

