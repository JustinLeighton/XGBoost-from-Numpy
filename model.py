import numpy as np
import math

class XGBoost():
    def __init__(self, maxDepth = 5,
                 learningRate = 0.3,
                 basePrediction = 0.5,
                 subSample = 1.0,
                 regLambda = 1.5,
                 gamma = 0.0,
                 minChildWeight = 1.0,
                 baseScore = 0.0,
                 colSampleByNode = 1.0,
                 seed=22):
        self.maxDepth = maxDepth
        self.learningRate = learningRate
        self.basePrediction = basePrediction
        self.subSample = subSample
        self.regLambda = regLambda
        self.gamma = gamma
        self.minChildWeight = minChildWeight
        self.baseScore = baseScore
        self.colSampleByNode = colSampleByNode
        self.rng = np.random.default_rng(seed=seed)
        
        # Unit tests
        assert self.colSampleByNode >= 0 and self.colSampleByNode <= 1, 'colSampleByNode must be between 0 and 1'
        assert self.maxDepth >= 0, 'max_depth must be nonnegative'
        
    def loss(self, y, pred): 
        return np.mean((y - pred)**2)
    
    def gradient(self, y, pred): 
        return np.array(pred - y)
    
    def hessian(self, y, pred): 
        return np.ones(len(y))
        
    def fit(self, X, y, numBoostRound, verbose=True):
        currentPredictions = self.basePrediction * np.ones(shape=y.shape)
        self.boosters = []
        
        for i in range(numBoostRound):
            gradients = self.gradient(y, currentPredictions)
            hessians = self.hessian(y, currentPredictions)
            
            sampleIdxs = None if self.subSample == 1.0 \
                else self.rng.choice(len(y), 
                                     size=math.floor(self.subSample*len(y)), 
                                     replace=False)
                
            # Column subsampling
            rand = self.rng.random(size=X.shape[1])
            mask = rand >= np.percentile(rand, (1 - self.colSampleByNode) * 100)

            # Generate booster                
            booster = TreeBooster(X=X, g=gradients, h=hessians, 
                                  maxDepth = self.maxDepth, 
                                  minChildWeight = self.minChildWeight,
                                  regLambda = self.regLambda,
                                  gamma = self.gamma,
                                  colMask = mask,
                                  idxs = sampleIdxs)
            currentPredictions += self.learningRate * booster.predict(X)
            self.boosters.append(booster)
            if verbose: 
                print(f'[{i}] train loss = {self.loss(y, currentPredictions)}')
            
    def predict(self, X):
        return (self.basePrediction + self.learningRate 
                * np.sum([booster.predict(X) for booster in self.boosters], axis=0))
    
class TreeBooster():
    def __init__(self, X, g, h, maxDepth, minChildWeight, regLambda, gamma, colMask, idxs=None):
        self.maxDepth = maxDepth
        self.minChildWeight = minChildWeight
        self.regLambda = regLambda
        self.gamma = gamma
        self.colMask = colMask
        if idxs is None: idxs = np.arange(len(g))
        self.X, self.g, self.h, self.idxs = X, g, h, idxs
        self.n, self.c = len(idxs), X.shape[1]
        
        # Equation 5 from XGBoost paper
        self.value = -g[idxs].sum() / (h[idxs].sum() + self.regLambda)
        self.topScore = 0.0
        if self.maxDepth > 0:
            self._maybeInsertChildNodes()

    def _maybeInsertChildNodes(self):
        for i in range(self.c): 
            if self.colMask[i]:
                self._findBetterSplit(i)
        if self.isLeaf(): return None
        x = self.X.values[self.idxs, self.splitFeatureIdx]
        leftIdx = np.nonzero(x <= self.threshold)[0]
        rightIdx = np.nonzero(x > self.threshold)[0]
        self.left = TreeBooster(self.X, self.g, self.h, self.maxDepth - 1, 
                                self.minChildWeight, self.regLambda, self.gamma, 
                                self.colMask, self.idxs[leftIdx])
        self.right = TreeBooster(self.X, self.g, self.h, self.maxDepth - 1, 
                                self.minChildWeight, self.regLambda, self.gamma, 
                                self.colMask, self.idxs[rightIdx])

    def isLeaf(self): 
        return self.topScore == 0.0
    
    def _findBetterSplit(self, featureIdx):
        x = self.X.values[self.idxs, featureIdx]
        g, h = self.g[self.idxs], self.h[self.idxs]
        sortIdx = np.argsort(x)
        sort_g, sort_h, sort_x = g[sortIdx], h[sortIdx], x[sortIdx]
        sum_g, sum_h = g.sum(), h.sum()
        sum_g_right, sum_h_right = sum_g, sum_h
        sum_g_left, sum_h_left = 0.0, 0.

        for i in range(0, self.n - 1):
            g_i, h_i, x_i, x_i_next = sort_g[i], sort_h[i], sort_x[i], sort_x[i + 1]
            sum_g_left += g_i; sum_g_right -= g_i
            sum_h_left += h_i; sum_h_right -= h_i
            if sum_h_left < self.minChildWeight or x_i == x_i_next:continue
            if sum_h_right < self.minChildWeight: break

            gain = 0.5 * ((sum_g_left**2 / (sum_h_left + self.regLambda))
                            + (sum_g_right**2 / (sum_h_right + self.regLambda))
                            - (sum_g**2 / (sum_h + self.regLambda))
                            ) - self.gamma/2 # Eq(7) in the xgboost paper
            if gain > self.topScore: 
                self.splitFeatureIdx = featureIdx
                self.topScore = gain
                self.threshold = (x_i + x_i_next) / 2
                
    def predict(self, X):
        return np.array([self._predict_row(row) for i, row in X.iterrows()])

    def _predict_row(self, row):
        if self.isLeaf(): 
            return self.value
        child = self.left if row[self.splitFeatureIdx] <= self.threshold else self.right
        return child._predict_row(row)
    

    
#%% Train
    
# Import data
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
X, y = fetch_california_housing(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=43)

# train the from-scratch XGBoost model
model = XGBoost(maxDepth = 5, colSampleByNode=0.75)
model.fit(X_train, y_train, 50)



#%% Pred

pred = model.predict(X_test)
print(pred)

def loss(y, pred): return np.mean((y - pred)**2)
print(f'Score: {loss(y_test, pred)}')




