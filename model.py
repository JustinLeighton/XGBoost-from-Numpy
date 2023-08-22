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
        
        # Input parameters
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
        
        # Input validation
        assert self.colSampleByNode >= 0, 'colSampleByNode must be greater than or equal to 0'
        assert self.colSampleByNode <= 1, 'colSampleByNode must be less than or equal to 1'
        assert self.subSample > 0, 'subSample must be greater than 0' 
        assert self.subSample <= 1, 'subSample must be less than or equal to 1'
        assert self.maxDepth >= 0, 'max_depth must be nonnegative'
    
    # Loss function for squared error objective
    def _loss(self, y, pred): 
        return np.mean((y - pred)**2)
    
    # Gradient function for squared error objective
    def _gradient(self, y, pred): 
        return np.array(pred - y)
    
    # Hessian function for squared error objective
    def _hessian(self, y, pred): 
        return np.ones(len(y))
    
    # Fit the boosting ensemble model to the training data
    def fit(self, X, y, numBoosters, verbose=True):
        
        # Initialize prediction array starting with basePrediction parameter.
        currentPredictions = self.basePrediction * np.ones(shape=y.shape)
        
        # Initialize booster array. This holds our base models.
        self.boosters = []
        
        # Create a number of base models equal to numBoosters.
        for i in range(numBoosters):
            
            # Calculate gradients and hessians besed upon currentPredictions
            gradients = self._gradient(y, currentPredictions)
            hessians = self._hessian(y, currentPredictions)
            
            # Row subsampling
            if self.subSample == 1.0:
                rowIdxs = None
            else:
                rowIdxs = self.rng.choice(len(y), size=math.floor(self.subSample*len(y)), replace=False)
                
            # Column subsampling by generating a mask on column headers
            rand = self.rng.random(size=X.shape[1])
            colMask = rand >= np.percentile(rand, (1 - self.colSampleByNode) * 100)

            # Fit base model on subsample
            booster = TreeBooster(X=X, g=gradients, h=hessians, 
                                  maxDepth = self.maxDepth, 
                                  minChildWeight = self.minChildWeight,
                                  regLambda = self.regLambda,
                                  gamma = self.gamma,
                                  colMask = colMask,
                                  rowIdxs = rowIdxs)
            
            # Update current predictions
            currentPredictions += self.learningRate * booster.predict(X)
            
            # Add base learner to ensemble
            self.boosters.append(booster)
            
            # Print loss to console
            if verbose: 
                print(f'[{i}] train loss = {self._loss(y, currentPredictions)}')
    
    # Calculate the prediction using the base model prediction,
    # learning rate, and the sum of predictions from individual boosters.
    # Each booster makes a prediction for the input data X, and the predictions
    # are aggregated using numpy.sum along axis 0.
    def predict(self, X):
        return (self.basePrediction + self.learningRate 
                * np.sum([booster.predict(X) for booster in self.boosters], axis=0))
    
    
    
class TreeBooster():
    def __init__(self, X, g, h, maxDepth, minChildWeight, regLambda, gamma, colMask, rowIdxs=None):
        
        # Input parameters
        self.maxDepth = maxDepth
        self.minChildWeight = minChildWeight
        self.regLambda = regLambda
        self.gamma = gamma
        self.colMask = colMask
        self.rowIdxs = rowIdxs
        if self.rowIdxs is None: self.rowIdxs = np.arange(len(g))
        self.X, self.g, self.h, self.idxs = X, g, h, self.rowIdxs
        self.n, self.c = len(self.rowIdxs), X.shape[1]
        self.topScore = 0.0
        
        # Calculate the value to update the leaf node during the tree boosting process.
        # The calculated value is used to update the leaf node during the gradient boosting process.
        # Found in equation #5 from the XGBoost paper
        self.value = -g[self.rowIdxs].sum() / (h[self.rowIdxs].sum() + self.regLambda)
        
        # Recursively build a binary tree structure by finding the best split rule for each node in the tree.
        if self.maxDepth > 0:
            self._insertChildNodes()

    # Check if the current node is a leaf node
    def isLeaf(self): 
        return self.topScore == 0.0

    # Insert child nodes for the current node during the tree building process.
    # This method is responsible for finding better splits for the current node's
    # features and creating left and right child nodes if a split improves the
    # quality of the tree.
    def _insertChildNodes(self):
        
        # Iterate over each feature (column) and attempt to find better splits.
        for i in range(self.c): 
            
            # Checks if column is masked from column subsampling
            if self.colMask[i]:
                
                # Attempt to find a better split
                self._findBetterSplit(i)
                
        # If the current node becomes a leaf node, no child nodes are inserted.
        if self.isLeaf(): return None
        
        # Get the values of the feature used for the split in the current node.
        x = self.X.values[self.rowIdxs, self.splitFeatureIdx]
        
        # Split the data into left and right based on the threshold.
        leftIdx = np.nonzero(x <= self.threshold)[0]
        rightIdx = np.nonzero(x > self.threshold)[0]
        
        # Create left and right child nodes with the appropriate subset of data.
        self.left = TreeBooster(self.X, self.g, self.h, self.maxDepth - 1, 
                                self.minChildWeight, self.regLambda, self.gamma, 
                                self.colMask, self.rowIdxs[leftIdx])
        self.right = TreeBooster(self.X, self.g, self.h, self.maxDepth - 1, 
                                self.minChildWeight, self.regLambda, self.gamma, 
                                self.colMask, self.rowIdxs[rightIdx])
    
    # Find a better split for a specific feature during the tree-building process
    # This function iterates through sorted values of a specific feature to find a split
    # that maximizes the gain in accordance with the XGBoost algorithm.
    # If a better split is found, the function updates the relevant variables in the node.
    def _findBetterSplit(self, featureIdx):
        
        # Get values of the selected feature for the samples in the current node
        x = self.X.values[self.rowIdxs, featureIdx]
        
        # Extract gradients and hessians for the selected samples.
        g, h = self.g[self.rowIdxs], self.h[self.rowIdxs]
        
        # Sort the values of the feature along with associated gradients and hessians.
        sortIdx = np.argsort(x)
        sort_g, sort_h, sort_x = g[sortIdx], h[sortIdx], x[sortIdx]
        
        # Initialize variables to track cumulative gradients and hessians.
        sum_g, sum_h = g.sum(), h.sum()
        sum_g_right, sum_h_right = sum_g, sum_h
        sum_g_left, sum_h_left = 0.0, 0.

        # Iterate through sorted values to find the best split.
        for i in range(0, self.n - 1):
            g_i, h_i, x_i, x_i_next = sort_g[i], sort_h[i], sort_x[i], sort_x[i + 1]
            sum_g_left += g_i; sum_g_right -= g_i
            sum_h_left += h_i; sum_h_right -= h_i
            
            # Check conditions for valid splits.
            if sum_h_left < self.minChildWeight or x_i == x_i_next: continue
            if sum_h_right < self.minChildWeight: break

            # Calculate gain using the XGBoost gain formula
            # Found in equation #7 in the XGBoost paper.
            gain = 0.5 * ((sum_g_left**2 / (sum_h_left + self.regLambda))
                            + (sum_g_right**2 / (sum_h_right + self.regLambda))
                            - (sum_g**2 / (sum_h + self.regLambda))
                            ) - self.gamma / 2
            
            # Update the best split if the calculated gain is higher.
            if gain > self.topScore: 
                self.splitFeatureIdx = featureIdx
                self.topScore = gain
                self.threshold = (x_i + x_i_next) / 2   
                
    # Predict each row of input
    def predict(self, X):
        return np.array([self._predict_row(row) for _, row in X.iterrows()])

    # Recursively predict the target value for a single input row by traversing the tree.
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




