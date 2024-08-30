class MyLinearBinaryRegressor:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = [0.0, 0.0]
        
    def fit(self, x, y):
        # sx0x0 = sum(x[i][0] * x[i][0] for i in range(len(x)))
        # sx1y = sum(x[i][1] * y[i] for i in range(len(x)))
        # sx0x1 = sum(x[i][0] * x[i][1] for i in range(len(x)))
        # sx0y = sum(x[i][0] * y[i] for i in range(len(x)))
        # sx1x1 = sum(x[i][1] * x[i][1] for i in range(len(x)))
        # sy = sum(y)
        # sx0 = sum(i[0] for i in x)
        # sx1 = sum(i[1] for i in x)
        
        # self.coef_[1] = (sx0x0 * sx1y - sx0x1 * sx0y) / (sx0x0 * sx1x1 - sx0x1 * sx0x1)
        # self.coef_[0] = (sx1x1 * sx0y - sx0x1 * sx1y) / (sx0x0 * sx1x1 - sx0x1 * sx0x1)
        # self.intercept_ = (sy - self.coef_[0] * sx0 - self.coef_[1] * sx1) / len(y)
        X = self.__create_matrix(x)
        Y = [val for val in y]
        toDelete = []
        for i in range(len(X)):
            for j in range(len(X[i+1:])):
                if X[i][1]*X[j][2] == X[i][2]*X[j][1]:
                    toDelete.append(j)
        toDelete = list(set(toDelete))
        for i in toDelete:
            del X[i]
            del Y[i]
            
            
        Xt = self.__transpose(X)
        B = self.__multiplyMatrixWithVector(self.__multiply(self.__inverse(self.__multiply(Xt, X)), Xt), Y)
        self.intercept_ = B[0]
        self.coef_[0] = B[1]
        self.coef_[1] = B[2]

    def predict(self, x):
        return [self.intercept_ + self.coef_[0] * val[0] + self.coef_[1] * val[1] for val in x]
    
    def __create_matrix(self, x):
        return [[1, val[0], val[1]] for val in x]
    
    def __transpose(self, matrix):
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    
    def __multiply(self, matrix1, matrix2):
        return [[sum(a * b for a, b in zip(row, col)) for col in zip(*matrix2)] for row in matrix1]
            
    def __minor(self, matrix, i, j):
        return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
    
    def __adjucate(self, matrix):
        new_matrix = self.__transpose(matrix)
        return [[(-1)**(i+j) * self.__calDet(self.__minor(new_matrix, i, j)) for j in range(len(new_matrix[0]))] for i in range(len(new_matrix))]
    
    def __calDet(self, matrix):
        if len(matrix) == 1:
            return matrix[0][0]
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        else:
            return sum((-1)**i * matrix[0][i] * self.__calDet(self.__minor(matrix, 0, i)) for i in range(len(matrix)))

    
    
    def __inverse(self, matrix):
        det = self.__calDet(matrix)
        while det == 0:
            print("Determinant 0")
        print(det)
        return [[val/det for val in row] for row in self.__adjucate(matrix)]
        
    
    def __multiplyMatrixWithVector(self, matrix, vector):
        return [sum(a * b for a, b in zip(row, vector)) for row in matrix]
    
    def test(self):
        A = [[3, 1, -6], [5, 2, -1], [-4, 3, 0]]
        B = [[1, 0, 2, 0], [3, 0, 0, 0], [2, 1, 4, 0], [1, 0, 5, 0]]
        print(self.__calDet(B))
        # print(self.__transpose(B))
        # print(self.__adjucate(B))
        # print(self.__inverse(B))
        # print(self.__multiply(B, self.__inverse(B)))
        