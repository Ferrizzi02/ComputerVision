from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    return 1/(1+np.exp(-x))

#derivative will derive the actual gradient
def sigmoid_deriv(x):
    return x*(1-x)

def predict(X, W):
    #f = W⋅xi
    preds = sigmoid_activation(X.dot(W))
    #step function so the ouput is binary
    preds[preds<= 0.5] = 0
    preds[preds > 0] = 1
    return preds

#parser
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default =100)
ap.add_argument("-a", "--alpha", type=float, default = 0.01)
args = vars(ap.parse_args())


'''test
x = np.arange(-10, 10, 0.1)
z = sigmoid_activation(x)
z2 = sigmoid_deriv(z)

plt.figure(figsize=(8,5))
plt.plot(x, z2)
plt.grid(True)
plt.show()
'''

#2 class clasification with 1000 points
# y is the original array of labels
# y.shape[0] is the number of samples
# reshape the array so it has 2Dim, where the second dim is 1
# Basically is making it a column vector instead of a row vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0],1))

#Insert a column of 1 (Bias trick) as the last entry in the feature matrix
'''
X = [[A B C]    =>  [[A B C 1]
     [D E F]]        [D E F 1]]

where x_0 = [A B C 1]^T
'''
X = np.c_[X, np.ones((X.shape[0]))]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize wheight matrix uniform distribution
W = np.random.rand(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args["epochs"]):
    #dot product between X and weight, it gives us our predictions
    preds = sigmoid_activation(trainX.dot(W))
    #error
    error = preds - trainY
    loss = np.sum(error**2)
    losses.append(loss)
    #gradient descent update is the dot product between features and error of
    #the sigmoid derivative of our prediction
    d = error * sigmoid_deriv(preds)
    gradient = trainX.T.dot(d) #transpuesta

    #update stage: nudge the weight in the negative direction of the gradient
    W += -args["alpha"] * gradient

    if epoch==0 or (epoch+1)%5==0:
        print("[FER:] epoch ={}, loss={:.7f}".format(int(epoch+1), loss))

print("[FER:] evaluating")
preds = predict(testX, W)
print(classification_report(testY, preds))

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()