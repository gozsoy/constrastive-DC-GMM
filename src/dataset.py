import numpy as np
import tensorflow as tf
import random
from scipy.sparse import csr_matrix
import scipy.sparse
import itertools


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

    def __init__(self, X, Y, alpha=1000, batch_size=100, num_constrains=0, q=0, ml=0, shuffle=True, l=0, feature_extractor=None):
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
        self.feature_extractor = feature_extractor

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


    def extract_image_features(self, batch_imgs):
        
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        '''batch_imgs = torch.from_numpy(batch_imgs.numpy())#.to(device)
        output = self.feature_extractor(batch_imgs)
        X = torch.sum(torch.sum(output,dim=-1),dim=-1)/9 # stl specific

        #if torch.cuda.is_available():
        #    X = X.data.cpu().numpy()
        #else:
        X = X.data.numpy()'''

        X = self.feature_extractor(batch_imgs, training=False)

        return X


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))
    
    def gen(self):
        while True:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.ind_constr)
            '''for index in range(int(len(self.X)/ self.batch_size)):
                indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
                X = tf.gather(self.X,indexes)
                X = self.extract_image_features(X) # added for stl10
                Y = self.Y[indexes]
                W = self.W[indexes][:, indexes]* self.alpha
                ind1, ind2 = csr_matrix_indices(W)
                data = W.data
                
                yield (X, (ind1, ind2, data)), {"output_1": X, "output_4": Y}'''
            for index in range(self.num_constrains// self.batch_size):
                indexes = self.ind_constr[index * self.batch_size//2:(index + 1) * self.batch_size//2]
                indexes = np.concatenate([self.ind1[indexes], self.ind2[indexes]])

                np.random.shuffle(indexes)
                X = tf.gather(self.X,indexes)
                X = self.extract_image_features(X) # added for stl10
                Y = self.Y[indexes]
                W = self.W[indexes][:, indexes]* self.alpha
                ind1, ind2 = csr_matrix_indices(W)
                data = W.data
                #W = W.toarray()
                yield (X, (ind1,ind2, data)), {"output_1": X, "output_4": Y}


class ContrastiveDataGenerator():
    'Generates data within contrastive learning schema for DC-GMM'

    def __init__(self, X, Y, alpha=1000, batch_size=100, feature_extractor=None):
        'Initialization'
        self.batch_size = batch_size
        self.alpha = alpha
        self.X = X
        self.Y = Y
        self.indexes = np.arange(len(self.Y))
        self.feature_extractor = feature_extractor
        
        self.ind1, self.ind2, self.data = self.get_batch_W()

        self.transformation1 = self.get_transformation_pipeline()
        self.transformation2 = self.get_transformation_pipeline()

    def extract_image_features(self, batch_imgs):

        X = self.feature_extractor(batch_imgs, training=False)

        return X

    def get_transformation_pipeline(self):
        
        transformation = tf.keras.Sequential([   
                tf.keras.layers.experimental.preprocessing.RandomCrop(height=48, width=72),
                tf.keras.layers.experimental.preprocessing.Resizing(height=96, width=96),
                tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
                tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1)
            ])

        return transformation


    def get_batch_W(self):

        all_pairs = list(itertools.product(range(self.batch_size), range(self.batch_size)))

        ind1 = []
        ind2 = []
        data = []

        def create_pair_relation(pair):
            el1 = pair[0]
            el2 = pair[1]

            if el1 == el2:
                ind1.append(el1)
                ind2.append(el2+self.batch_size)
                data.append(1.)

                ind1.append(el2+self.batch_size)
                ind2.append(el1)
                data.append(1.)
            
            else:
                ind1.append(el1)
                ind2.append(el2)
                data.append(-1.)

                ind1.append(el1)
                ind2.append(el2+self.batch_size)
                data.append(-1.)

                ind1.append(el2+self.batch_size)
                ind2.append(el1)
                data.append(-1.)

                # do not forget to add
                # el1+batch_size, el2+batch_size

            return

        _ = list(map(create_pair_relation,all_pairs))

        ind1 = np.array(ind1, dtype=np.int32)
        ind2 = np.array(ind2, dtype=np.int32)
        data = np.array(data) * self.alpha

        return ind1, ind2, data


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))
    
    def gen(self):
        while True:
            np.random.shuffle(self.indexes)
            for index in range(int(len(self.X)/ self.batch_size)):
                indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

                X = tf.gather(self.X,indexes)

                # TODO: test data will not have below
                X_aug1 = self.transformation1(X, training=True)
                X_aug2 = self.transformation2(X, training=True)
                X = tf.concat((X_aug1, X_aug2),axis=0)


                X = self.extract_image_features(X) # (N,96,96,3) -> (N,2048)
                Y = tf.concat((self.Y[indexes],self.Y[indexes]),axis=0)
                
                yield (X, (self.ind1, self.ind2, self.data)), {"output_1": X, "output_4": Y}
