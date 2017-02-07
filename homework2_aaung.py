# NOTE -- please do NOT put your name(s) in the Python code; instead, name the Python file
# itself to include your WPI username(s).

import cv2  # Uncomment if you have OpenCV and want to run the real-time demo
import numpy as np
import matplotlib.pyplot as plt

def generateEigVectors(faces):
    samples  = faces.shape[0]
    dimensions = faces.shape[1]
    vecMean = np.mean(faces, axis=0)
    matMean = np.tile(vecMean,samples).reshape(samples,dimensions)
    faces = faces - matMean
    cov = np.cov(faces.T)
    eigValues, eigVectors = np.linalg.eig(cov)
    idx = eigValues.argsort()[::-1]   
    eigValues  = eigValues[idx]
    eigVectors  = eigVectors[:,idx]
    #eigVectors[:, eigValues.argmax()]
    return eigVectors


def J(w, faces, labels, alpha=0.):
    x = faces
    y = labels
    loss = x.dot(w) - y
    ridge = alpha * w.dot(w)
    cost = 0.5 * (loss.dot(loss) + ridge)
    return cost


def gradJ(w, faces, labels, alpha=0.):
    x = faces
    y = labels
    loss = x.dot(w) - y
    grad = x.T.dot(loss) + (alpha * w)
    return grad


def gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha=0.):
    x = trainingFaces
    y = trainingLabels
    epsilon =  7 * 1e-6
    delta = 1e-5

    w = np.random.rand(x.shape[1])
    cost1 = J(w, x, y)
    newgrad = gradJ(w, x, y, alpha)
    w = w - (epsilon* newgrad)
    cost2 = J(w, x, y)

    while abs(cost1 - cost2) > delta:
        cost1 = J(w, x, y)
        newgrad = gradJ(w, x, y, alpha)
        w = w - (epsilon * newgrad)
        
        cost2 = J(w, x, y)
        print "Cost: ", cost2, "|w|:", w.dot(w)
    return w


def method1(trainingFaces, trainingLabels, testingFaces, testingLabels):
    x = trainingFaces
    y = trainingLabels
    A = np.dot(x.T, x)
    B = np.dot(x.T, y)
    w = np.linalg.solve(A, B)
    print "|w|: ", w.dot(w)
    return w


def method2(trainingFaces, trainingLabels, testingFaces, testingLabels):
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels)


def method3(trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 1e3
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha)


def reportCosts(w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha=0.):
    print "Training cost: {}".format(J(w, trainingFaces, trainingLabels, alpha))
    print "Testing cost:  {}".format(J(w, testingFaces, testingLabels, alpha))


# Accesses the web camera, displays a window showing the face, and classifies smiles in real time
# Requires OpenCV.
def detectSmiles(w):
    # Given the image captured from the web camera, classify the smile
    def classifySmile(im, imGray, faceBox, w):
        # Extract face patch as vector
        face = imGray[faceBox[1]:faceBox[1] + faceBox[3], faceBox[0]:faceBox[0] + faceBox[2]]
        face = cv2.resize(face, (24, 24))
        face = (face - np.mean(face)) / np.std(face)  # Normalize
        face = np.reshape(face, face.shape[0] * face.shape[1])

        # Classify face patch
        yhat = w.dot(face)
        # print yhat

        # Draw result as colored rectangle
        THICKNESS = 3
        green = 128 + (yhat - 0.5) * 255
        color = (0, green, 255 - green)
        pt1 = (faceBox[0], faceBox[1])
        pt2 = (faceBox[0] + faceBox[2], faceBox[1] + faceBox[3])
        cv2.rectangle(im, pt1, pt2, color, THICKNESS)
        cv2.putText(im, str(yhat), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    # Starting video capture
    vc = cv2.VideoCapture()
    vc.open(0)
    faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")  # TODO update the path
    while vc.grab():
        (tf, im) = vc.read()
        im = cv2.resize(im, (im.shape[1] / 2, im.shape[0] / 2))  # Divide resolution by 2 for speed
        imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        k = cv2.waitKey(30)
        if k >= 0 and chr(k) == 'q':
            print "quitting"
            break

        # Detect faces
        faceBoxes = faceDetector.detectMultiScale(imGray)
        for faceBox in faceBoxes:
            classifySmile(im, imGray, faceBox, w)
        cv2.imshow("WebCam", im)

    cv2.destroyWindow("WebCam")
    vc.release()


if __name__ == "__main__":
    # Load data
    if ('trainingFaces' not in globals()):  # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
        trainingFaces = np.load("trainingFaces.npy")
        trainingLabels = np.load("trainingLabels.npy")
        testingFaces = np.load("testingFaces.npy")
        testingLabels = np.load("testingLabels.npy")


    eigVectors = generateEigVectors(trainingFaces)
    reEig = np.reshape(eigVectors[0], (24,24))
    print (eigVectors)
#    w1 = method1(trainingFaces, trainingLabels, testingFaces, testingLabels)
#    w2 = method2(trainingFaces, trainingLabels, testingFaces, testingLabels)
#    w3 = method3(trainingFaces, trainingLabels, testingFaces, testingLabels)
#
#    for w in [w1, w2, w3]:
#        reportCosts(w, trainingFaces, trainingLabels, testingFaces, testingLabels)
#
#    w_shaped = np.reshape(w3, (24, 24))
#    img = plt.imshow(w_shaped, cmap='gray')
#    plt.show(img)
    #detectSmiles(w3)  # Requires OpenCV
