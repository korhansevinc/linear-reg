# linear regression code

def get_shape(matrix):
    num_samples = len(matrix)
    num_features = len(matrix[0]) if len(matrix) != 0 else 0
    return num_samples, num_features

def initiliaze_weights(num_features):
    weights = []
    for i in range(num_features):
        weights.append(0)
    #print(weights)
    return weights

def calculate_y_predictions(X, weights, bias):
    y_pred= []
    num_samples, num_features = get_shape(X)
    for i in range(num_samples) :
        val = X[i][0] * weights[0] + X[i][1] * weights[1] + bias
        y_pred.append(val)
    return y_pred

class LinearRegression:
    
    def __init__(self, learning_rate = 0.01, epoch=1000) :
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weights = [0,0] # weights for X1 and X2 [ [0,0] [0,1] ... ]
        self.bias = 0 # b
        self.Target_Tests_in_epochs = [[]]
        self.Target_Trains_in_epochs = [[]]

    def fit(self, X, y, X_test):
        num_samples, num_features = get_shape(X) # num_samples: kac tane 2 li var, num_features : her sample kac kisi.
        self.bias = 0
        
        for _ in range(self.epoch) :
            y_pred = calculate_y_predictions(X, self.weights, self.bias)
            sum1 = 0
            sum2 = 0
            sum3 = 0
            for i in range(num_samples) :
                val1 = (y_pred[i] - y[i][0]) * X[i][0]
                val2 = (y_pred[i] - y[i][0]) * X[i][1]
                val3 = (y_pred[i] - y[i][0])
                sum1 = sum1 + val1
                sum2 = sum2 + val2
                sum3 = sum3 + val3
            
            dw1 = (2/num_samples) * sum1
            dw2 = (2/num_samples) * sum2
            db  = (2/num_samples) * sum3
            
            self.weights[0] = self.weights[0] - self.learning_rate * dw1
            self.weights[1] = self.weights[1] - self.learning_rate * dw2
            self.bias = self.bias - self.learning_rate * db

            Target_Train = calculate_y_predictions(X, self.weights, self.bias) #her epoch'un sonunda hesaplanmis target val : train set icin
            self.Target_Trains_in_epochs.append(Target_Train)
            Target_Test = calculate_y_predictions(X_test, self.weights, self.bias) #her epoch'un sonunda hesaplanmis target val : test set icin
            self.Target_Tests_in_epochs.append(Target_Test) # y_test_predict degerleri her bir epoch icin

    def predict(self, X):
        y_pred = calculate_y_predictions(X, self.weights, self.bias)
        return y_pred
