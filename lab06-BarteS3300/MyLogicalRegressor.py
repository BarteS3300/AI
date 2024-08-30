import numpy as np

class MyLogicalRegressorFunction():
    def __init__(self, y, learning_rate = 0.01, n_iterations = 1000,):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.intercept_ = [0.0]
        self.coef_ = []
        self.result = y
        
    def fitBinary(self, x, y):
        self.coef_ = [0.0] * len(x[0])
        self.dif_outputs = list(set(y))
        
        for epoch in range(self.n_iterations):
            for i in range(len(x)):
                error = self.sigmoid(x[i]) - y[i]
                
                self.intercept_[0] += self.learning_rate * error
                
                for j in range(len(x[i])):
                    self.coef_[j] += self.learning_rate * error * x[i][j]
        if epoch > 0 and epoch % 10 == 0:
            self.learning_rate = self.learning_rate * 0.5

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-(self.intercept_[0] - sum([self.coef_[j] * x[j] for j in range(len(x))]))))
    
    def predict(self, x):
        return self.sigmoid(x)

class MyLogicalRegressor():
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.intercept_ = [0.0]
        self.coef_ = []
        self.functions = []
    
    def fitBinary(self, x, y):
        self.coef_ = [0.0] * len(x[0])
        self.dif_outputs = list(set(y))
        
        for epoch in range(self.n_iterations):
            for i in range(len(x)):
                error = self.sigmoid(x[i]) - y[i]
                
                self.intercept_[0] += self.learning_rate * error
                
                for j in range(len(x[i])):
                    self.coef_[j] += self.learning_rate * error * x[i][j]
        if epoch > 0 and epoch % 10 == 0:
            self.learning_rate = self.learning_rate * 0.5
        
    
    def fit(self, x, y):
        self.coef_ = [0.0] * len(x[0])
        self.dif_outputs = list(set(y))
        if(len(self.dif_outputs) == 2):
            for epoch in range(self.n_iterations):
                for i in range(len(x)):
                    error = self.sigmoid(x[i]) - y[i]
                    
                    self.intercept_[0] += self.learning_rate * error
                    
                    for j in range(len(x[i])):
                        self.coef_[j] += self.learning_rate * error * x[i][j]
            if epoch > 0 and epoch % 10 == 0:
                self.learning_rate = self.learning_rate * 0.5
        
        else:
            ys = []
            for val in self.dif_outputs:
                ys.append([1 if val == y[i] else 0 for i in range(len(y))])
            self.functions = [MyLogicalRegressorFunction(j, self.learning_rate, self.n_iterations) for j in self.dif_outputs]
            for i in range(len(self.functions)):
                self.functions[i].fitBinary(x, ys[i])
                    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-(self.intercept_[0] - sum([self.coef_[j] * x[j] for j in range(len(x))]))))
    
    def predict(self, x):
        result = []
        if(len(self.dif_outputs) == 2):
            for i in range(len(x)):
                sigmoid = (self.sigmoid(x[i]))
                result.append(1 if sigmoid >= 0.5 else 0)
            return result
        else:
            ys = [f.result for f in self.functions]
            for i in range(len(x)):
                dict = {ys[j] : self.functions[j].predict(x[i]) for j in range(len(ys))}
                result.append(max(dict, key=dict.get))
            return result