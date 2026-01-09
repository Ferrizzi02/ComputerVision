#Still not training :( but building the parameters
import numpy as np
import cv2

labels = ["dog", "cat", "panda"]
np.random.seed(1) #random number

#initialize weigt matrix and buas vectir
W = np.random.randn(3,3072) #32*32*3
b = np.random.randn(3) #3 labels

foto = cv2.imread("Z_02_LINEAR/beagle.png")
foto_resized = cv2.resize(foto, (32,32)).flatten()

scores = W.dot(foto_resized)+b #score function

for (label, score) in zip(labels, scores):
    print("Label: {}: {:.2f}".format(label, score))


cv2.putText(foto, "Label: {}".format(labels[np.argmax(scores)]),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

cv2.imshow("Image", foto)
cv2.waitKey(0)
