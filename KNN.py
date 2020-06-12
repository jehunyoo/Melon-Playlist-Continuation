import numpy as np
import pickle
from tqdm.notebook import tqdm # notebook only

class KNN:

    def __init__(self, k, metric="euclidean", weights="uniform", verbose=True, save=False, save_as=None):
        self.X = None # X_train
        self.y = None # y_train
        self._dists = None
        self.k = k
        self.metric = metric
        self.weights = weights
        self.verbose = verbose
        self.save = save
        self.save_as = save_as

    def fit(self, X, y):
        '''
        X: pandas.DataFrame
        y: pandas.Series
        '''
        self.X = X.to_numpy().copy()
        self.y = y.to_numpy().copy()
        self._dists = None # renew _dists; changed X and y

    def predict(self, X_test=None):
        '''
        X_test: pandas.DataFrame
        '''
        if self._dists and (not X_test):
            dists = np.argsort(self._dists)[:, :self.k]
        if self.X.shape[1] != X_test.shape[1]:
            raise "columns of X_train and columns of X_test are different."
        else:
            self.dists = None # memory management
            dists = self._distance(X_test) # dists.shape = (X_test.shape[0], self.k)
        label = np.take(self.y, dists); del dists # label.shape - (X_test.shape[0], self.k)
        y_pred = self._weight(label)
        return y_pred

    def _distance(self, X_test):
        '''
        dists: shape = (X_test.shape[0], X_train.shape[0])
        '''
        dists = np.empty([X_test.shape[0], self.X.shape[0]])
        X_test = X_test.to_numpy()

        print("calculate distance ...") if self.verbose else None
        _range = tqdm(range(X_test.shape[0])) if self.verbose else range(X_test.shape[0])

        if self.metric == "euclidean": # numpy broadcasting
            for idx in _range:
                dists[idx, :] = np.sqrt(np.sum((X_test[idx, :] - self.X) ** 2, axis=1))
        elif self.metric == "manhattan":
            for idx in _range:
                dists[idx, :] = np.sum(np.abs(X_test[idx, :] - self.X), axis=1)
        # {{ add other distances here }}
        else:
            raise KeyError("{} is not found.".format(self.metric))
        
        try:
            self._save(dists) if self.save else None # save dists
        except:
            print("Error: {} does not saved.".format(self.save_as))
        self._dists = dists

        return np.argsort(dists)[:, :self.k]

    def _weight(self, label):
        y_pred = np.empty(label.shape[0])

        print("calculate labels ...") if self.verbose else None
        _range = tqdm(range(label.shape[0])) if self.verbose else range(label.shape[0])

        if self.weights == "uniform":
            # count labels and return predicted numpy array
            for idx in _range:
                u, c = np.unique( label[idx, :], return_counts=True )
                y_pred[idx] = u[np.argmax(c)]
        elif self.weights == "distance":
            pass
        # {{ add other weights here }}
        else:
            raise KeyError("{} is not found".format(self.weights))
        return y_pred
    
    def _save(self, data):
        with open(self.save_as, 'wb') as f:
            pickle.dump(data, f)
            print("saved at {}".format(self.save_as))

    def load(self, load_as):
        '''
        load distance matrix
        '''
        with open(load_as, 'rb') as f:
            self._dists = pickle.load(f)



if __name__=="__main__": # test
    import pandas as pd
    X = pd.DataFrame([[0,1,2,3,4,34,35],[5,6,7,8,9,21,11], [9,8,7,6,5,4,3]]) # three data
    y = pd.Series([0,1,0]) # labels
    X_test = pd.DataFrame([[4,6,32,5,7,3,8]]) # check for 1 data
    knn = KNN(k=3)
    knn.fit(X,y)
    y_pred = knn.predict(X_test)
    print(y_pred) # y_pred should be 0 because k is 3.