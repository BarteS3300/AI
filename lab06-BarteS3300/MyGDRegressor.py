from cmath import exp


class MyGDRegressor:
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.intercept_ = [0.0]
        self.coef_ = []

    def fitBatch1Param(self, x, y):
        start_lr = self.learning_rate
        self.coef_.append(0.0)
        for epoch in range(self.n_iterations):
            predictions = self.predict(x)
            
            # Calculate the gradient for the intercept
            slope_intercept = sum(-2 *(predictions[i] - y[i]) for i in range(len(y))) / len(y)
            step_intercept = slope_intercept * self.learning_rate
            self.intercept_[0] -= step_intercept
            
            # Calculate the gradient for the coefficient
            slope_coef = sum(-2 * x[i][0] * (y[i] - predictions[i]) for i in range(len(y))) / len(y)
            step_coef = slope_coef * self.learning_rate
            
            self.intercept_[0] -= step_intercept
            self.coef_[0] -= step_coef
            
            if(epoch > 0 and epoch % 16 == 0):
                self.learning_rate = self.learning_rate * 0.7945
        
        self.learning_rate = start_lr
            
            #Stope the training if the error is small enough
            # if abs(step_intercept) < 0.0001 and abs(step_coef) < 0.0001:
            #     break

    #Sum of squared residuals
    def fitBatch2Params(self, x, y):
        start_lr = self.learning_rate
        self.coef_ = [0.0] * 2
        for epoch in range(self.n_iterations):
            predictions = self.predict(x)
            
            # Calculate the gradient for the intercept
            slope_intercept = sum(-2*(predictions[i] - y[i]) for i in range(len(y)))
            step_intercept = slope_intercept * self.learning_rate
            self.intercept_[0] -= step_intercept
            
            #Calculate the gradient for the first coefficient
            slope_coef1 = sum(-2* x[i][0] * (y[i] - predictions[i]) for i in range(len(y)))
            step_coef1 = slope_coef1 * self.learning_rate
            self.coef_[0] -= step_coef1
            
            # Calculate the gradient for the second coefficient
            slope_coef2 = sum(-2 * x[i][1] * (y[i] - predictions[i]) for i in range(len(y)))
            step_coef2 = slope_coef2 * self.learning_rate
            self.coef_[1] -= step_coef2
            
            if(epoch > 0 and epoch % 16 == 0):
                self.learning_rate = self.learning_rate * 0.7945
            
            #Stope the training if the error is small enough
            # if abs(step_intercept) < 0.0001 and abs(step_coef1) < 0.0001 and abs(step_coef2) < 0.0001:
            #     break
        self.learning_rate = start_lr
    
    def fitBatch(self, x, y):
        self.coef_ = [0.0] * len(x[0])
        start_lr = self.learning_rate
        for epoch in range(self.n_iterations):
            predictions = self.predict(x)
        
        # Calculate the gradient for the intercept
            slope_intercept = sum(-1*(predictions[i] - y[i]) for i in range(len(y))) / len(y)
            step_intercept = slope_intercept * self.learning_rate
            self.intercept_[0] -= step_intercept

            slope_coef = [None] * len(x[0])
            step_coef = [None] * len(x[0])
            # Calculate the gradient for the coefficients
            for i in range(len(x[0])):
                slope_coef[i] = sum(-1 * x[j][i] * (y[j] - predictions[j]) for j in range(len(y))) / len(y)
                step_coef[i] = slope_coef[i] * self.learning_rate
                self.coef_[i] -= step_coef[i]
            
            if(epoch > 0 and epoch % 16 == 0):
                self.learning_rate = self.learning_rate * 0.7945
            
        self.learning_rate = start_lr
        
    
    def predict(self, x):
        return [self.intercept_[0] + sum(self.coef_[i] * val[i] for i in range(len(val))) for val in x]
    
    