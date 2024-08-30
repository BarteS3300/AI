import numpy as np

class MyNeuralNetwork:
    def __init__(self, m, n):
        self.W1 = np.random.rand(128, n) - 0.5
        self.b1 = np.random.rand(128, 1) - 0.5
        self.W2 = np.random.rand(1, 128) - 0.5
        self.b2 = np.random.rand(1, 1) - 0.5
        # print("W1: ", self.W1)
        # print("b1: ", self.b1)
        # print("W2: ", self.W2)
        # print("b2: ", self.b2)
        
    def flatten(self, mat):
        x =[]
        for line in mat:
            for el in line:
                x.append(el)
        return x
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def sigmoid(self, Z):
        Zclipped = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Zclipped))
    
    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.relu(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.sigmoid(Z2)
        return Z1, A1, Z2, A2
    
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max()+1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
    
    def one_hot_sigmoid(self, Y):
        one_hot_Y = Y.reshape(1, Y.size)
        return one_hot_Y
    
    def derivateReLU(self, Z):
        return Z > 0
        
    def back_prop(self, Z1, A1, Z2, A2, X, Y):
        m = Y.size
        one_hot_Y = self.one_hot_sigmoid(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1/m * dZ2.dot(A1.T)
        db2 = 1/m * np.sum(dZ2)
        dZ1 = self.W2.T.dot(dZ2) * self.derivateReLU(Z1)
        dW1 = 1/m * dZ1.dot(X.T)
        db1 = 1/m * np.sum(dZ1)
        return dW1, db1, dW2, db2
    
    def update_params(self, dW1, db1, dW2, db2, alpha):
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2
        
    def getPredict(A2):
        return np.argmax(0, A2)
        
    def getPredictSigmoid(self, A2):
        predictions = []
        for val in A2[0]:
            if val > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        predictions = np.array(predictions)
        predictions = predictions.reshape(1, predictions.size)
        return predictions
    
    def accuracy(self, predictions, Y):
        # print(self.W1, self.b1, self.W2, self.b2)
        # print(predictions, Y)
        return np.sum(predictions == Y) / Y.size
        
    def gradient_descent(self, X, Y, alpha, epochs):
        for i in range(epochs):
            Z1, A1, Z2, A2 = self.forward_prop(X)
            dW1, db1, dW2, db2 = self.back_prop(Z1, A1, Z2, A2, X, Y)
            self.update_params(dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                print('Epoch: ', i, 'Accuracy: ', self.accuracy(self.getPredictSigmoid(A2), Y))
    
    def fit(self, X, Y, alpha, epochs):
        self.gradient_descent(X, Y, alpha, epochs)
        
    def predict(self, X):
        _, _, _, A2 = self.forward_prop(X)
        return A2[0]