
# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# to compare our model's accuracy with sklearn model
from sklearn.linear_model import LogisticRegression

"""
def split(x, y):
    z = mt.floor(x.shape[1]*0.8)
    a = x.shape[1] - z
    x_train = x[:, :z]
    x_test = x[:, z:]

    y_train = y[:, :z]
    y_test = y[:, z:]
    return x_train, x_test, y_train, y_test
"""
def Normalize(X):
    _min = X.min(axis=0)
    _max = X.max(axis=0)
    z = np.divide(np.subtract(X, _min), np.subtract(_max, _min))
    return z

# Logistic Regression
class LogicRegression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weight = None
        self.bias = None


    def fit (self,X,Y):
     n_samples , n_features = X.shape
     self.weight=np.zeros(n_features)
     self.bias=0
     costList= []
     #gredientdescent
     for i in range (self.iterations):
         #WX+=b
         linear_hx = np.dot(X,self.weight) + self.bias
         y_predict = self.sigmoid(linear_hx)

         #update the weight and bias
         dw = (1/n_samples) * np.dot(X.T,(y_predict - Y))
         db = (1/n_samples) * np.sum(y_predict - Y)

         self.weight -= self.learning_rate *dw
         self.bias -= self.learning_rate *db
         self.cost_function(y_predict,Y)
         cost = self.cost_function(y_predict,Y)
         costList.append(cost)

         if (i % (self.iterations / 10) == 0):
             print('cost after ', i, " iterations is ", cost)
     return costList


    def cost_function(self,A, Y):

        m = Y.shape[0]
        cost = -(1/m) * np.sum((Y*np.log(A)) + ((1-Y)*np.log(1-A)))
        return cost

    def sigmoid (self , z):
        return (1/ (1+ np.exp(-z)))



    def  predict (self,X):
        Z = 1 / (1 + np.exp(- (X.dot(self.weight) + self.bias)))
        y = np.where(Z > 0.5, 1, 0)
        return y



# Driver code

def main():
    # Importing dataset
    data = pd.read_csv('heart.csv')
    feature_cols = ['trestbps', 'chol', 'thalach', 'oldpeak']
    X = data[feature_cols].values  # Features
    Y = data['target'].values  # output
    X = Normalize(X)

    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)
    iterations =15000

    # Model training
    model = LogicRegression(learning_rate=0.1, iterations=15000)
    costList =model.fit(X_train, Y_train)
    model1 = LogisticRegression()
    model1.fit(X_train, Y_train)
    plt.plot(np.arange(iterations), costList)
    plt.show()

    # Prediction on test set
    Y_pred = model.predict(X_test)


    # measure performance
    correctly_classified = 0
    accuaracy_skmodel = model1.score(X_test, Y_test)
    error_skmodel = 1 - accuaracy_skmodel

    # counter
    count = 0
    for count in range(np.size(Y_pred)):

        if Y_test[count] == Y_pred[count]:
            correctly_classified = correctly_classified + 1

        count = count + 1
    acc = correctly_classified / count *100
    print("Accuracy on test set by our model       :  ", acc )
    print("Accuracy on test set by sklearn model   :  ", (accuaracy_skmodel *100))
    print("Error in sklearn model",error_skmodel*100)
    print("Error by our model",(100-acc) )



if __name__ == "__main__":
    main()