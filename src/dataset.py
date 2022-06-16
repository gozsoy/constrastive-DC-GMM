import numpy as np
import tensorflow as tf
import random
from scipy.sparse import csr_matrix
import scipy.sparse


def csr_matrix_indices(S):
    """
    Return a list of the indices of nonzero entries of a csr_matrix S
    """
    major_dim, minor_dim = S.shape
    minor_indices = S.indices

    major_indices = np.empty(len(minor_indices), dtype=S.indices.dtype)
    scipy.sparse._sparsetools.expandptr(major_dim, S.indptr, major_indices)

    return major_indices, minor_indices

class DataGenerator():
    'Generates data for Keras'

    def __init__(self, X, Y, alpha=1000, batch_size=100, num_constrains=0, q=0, ml=0, shuffle=True, l=0):
        'Initialization'
        self.batch_size = batch_size
        self.alpha = alpha
        self.q = q
        self.num_constrains = num_constrains
        self.ml = ml
        self.X = X
        if l == 0:
            self.l = len(Y)
        else:
            self.l = l
        self.Y = Y
        self.W, self.ml_ind1, self.ml_ind2, self.cl_ind1, self.cl_ind2 = self.get_W()
        self.ind1 = np.concatenate([self.ml_ind1,self.cl_ind1])
        self.ind2 = np.concatenate([self.ml_ind2,self.cl_ind2])
        self.indexes = np.arange(len(self.Y))
        self.ind_constr = np.arange(self.num_constrains)
        self.shuffle = shuffle

    def transitive_closure(self, ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
        """
        This function calculate the total transtive closure for must-links and the full entailment
        for cannot-links.

        # Arguments
            ml_ind1, ml_ind2 = instances within a pair of must-link constraints
            cl_ind1, cl_ind2 = instances within a pair of cannot-link constraints
            n = total training instance number
        # Return
            transtive closure (must-links)
            entailment of cannot-links
        """
        ml_graph = dict()
        cl_graph = dict()
        for i in range(n):
            ml_graph[i] = set()
            cl_graph[i] = set()

        def add_both(d, i, j):
            d[i].add(j)
            d[j].add(i)

        for (i, j) in zip(ml_ind1, ml_ind2):
            add_both(ml_graph, i, j)

        def dfs(i, graph, visited, component):
            visited[i] = True
            for j in graph[i]:
                if not visited[j]:
                    dfs(j, graph, visited, component)
            component.append(i)

        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, ml_graph, visited, component)
                for x1 in component:
                    for x2 in component:
                        if x1 != x2:
                            ml_graph[x1].add(x2)
        for (i, j) in zip(cl_ind1, cl_ind2):
            add_both(cl_graph, i, j)
            for y in ml_graph[j]:
                add_both(cl_graph, i, y)
            for x in ml_graph[i]:
                add_both(cl_graph, x, j)
                for y in ml_graph[j]:
                    add_both(cl_graph, x, y)
        ml_res_set = set()
        cl_res_set = set()
        for i in ml_graph:
            for j in ml_graph[i]:
                if j != i and j in cl_graph[i]:
                    raise Exception('inconsistent constraints between %d and %d' % (i, j))
                if i <= j:
                    ml_res_set.add((i, j))
                else:
                    ml_res_set.add((j, i))
        for i in cl_graph:
            for j in cl_graph[i]:
                if i <= j:
                    cl_res_set.add((i, j))
                else:
                    cl_res_set.add((j, i))
        ml_res1, ml_res2 = [], []
        cl_res1, cl_res2 = [], []
        for (x, y) in ml_res_set:
            ml_res1.append(x)
            ml_res2.append(y)
        for (x, y) in cl_res_set:
            cl_res1.append(x)
            cl_res2.append(y)
        return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)

    def generate_random_pair(self, y, num, q):
        """
        Generate random pairwise constraints.
        """
        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []
        while num > 0:
            tmp1 = random.randint(0, self.l - 1)
            tmp2 = random.randint(0, self.l - 1)
            ii = np.random.uniform(0,1)
            if tmp1 == tmp2:
                continue
            if y[tmp1] == y[tmp2]:
                if ii >= q:
                    ml_ind1.append(tmp1)
                    ml_ind2.append(tmp2)
                else:
                    cl_ind1.append(tmp1)
                    cl_ind2.append(tmp2)
            else:
                if ii >= q:
                    cl_ind1.append(tmp1)
                    cl_ind2.append(tmp2)
                else:
                    ml_ind1.append(tmp1)
                    ml_ind2.append(tmp2)
            num -= 1
        return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)

    def generate_random_pair_ml(self, y, num):
        """
        Generate random pairwise constraints.
        """
        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []
        while num > 0:
            tmp1 = random.randint(0, y.shape[0] - 1)
            tmp2 = random.randint(0, y.shape[0] - 1)
            ii = np.random.uniform(0,1)
            if tmp1 == tmp2:
                continue
            if y[tmp1] == y[tmp2]:
                ml_ind1.append(tmp1)
                ml_ind2.append(tmp2)
                num -= 1
        return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)

    def generate_random_pair_cl(self, y, num):
        """
        Generate random pairwise constraints.
        """
        ml_ind1, ml_ind2 = [], []
        cl_ind1, cl_ind2 = [], []
        while num > 0:
            tmp1 = random.randint(0, y.shape[0] - 1)
            tmp2 = random.randint(0, y.shape[0] - 1)
            ii = np.random.uniform(0,1)
            if tmp1 == tmp2:
                continue
            if y[tmp1] != y[tmp2]:
                cl_ind1.append(tmp1)
                cl_ind2.append(tmp2)
                num -= 1
        return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)

    def get_W(self):
        if self.ml==0:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.generate_random_pair(self.Y, self.num_constrains, self.q)
            if self.q == 0:
                ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, self.X.shape[0])
        elif self.ml == 1:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.generate_random_pair_ml(self.Y, self.num_constrains)
        elif self.ml == -1:
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = self.generate_random_pair_cl(self.Y, self.num_constrains)
        print("\nNumber of ml constraints: %d, cl constraints: %d.\n " % (len(ml_ind1), len(cl_ind1)))
        
        ind1 = np.concatenate([ml_ind1, ml_ind2, cl_ind1, cl_ind2])
        ind2 = np.concatenate([ml_ind2, ml_ind1, cl_ind2, cl_ind1])
        data = np.concatenate([np.ones(len(ml_ind1)*2), np.ones(len(cl_ind1)*2)*-1])
        W = csr_matrix((data, (ind1, ind2)), shape=(len(self.X), len(self.X)))
        W = W.tanh().rint()
        return W, ml_ind1, ml_ind2, cl_ind1, cl_ind2

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))
    
    def gen(self):
        while True:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.ind_constr)
            for index in range(int(len(self.X)/ self.batch_size)):
                indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
                X = self.X[indexes]
                Y = self.Y[indexes]
                W = self.W[indexes][:, indexes]* self.alpha
                ind1, ind2 = csr_matrix_indices(W)
                data = W.data
                yield (X, (ind1, ind2, data)), {"output_1": X, "output_4": Y}
            for index in range(self.num_constrains// self.batch_size):
                indexes = self.ind_constr[index * self.batch_size//2:(index + 1) * self.batch_size//2]
                indexes = np.concatenate([self.ind1[indexes], self.ind2[indexes]])
                np.random.shuffle(indexes)
                X = self.X[indexes]
                Y = self.Y[indexes]
                W = self.W[indexes][:, indexes]* self.alpha
                ind1, ind2 = csr_matrix_indices(W)
                data = W.data
                #W = W.toarray()
                yield (X, (ind1,ind2, data)), {"output_1": X, "output_4": Y}
