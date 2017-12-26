import unittest

from numpy import array

from knn import *
import matplotlib.pyplot as plt

# import data
data = Numbers("data/mnist.pkl.gz")

done1 = 0
done2 = 0
done2b = 0
done3 = 1
done4 = 1

# 1.2: plot accuracy vs training instances
if done1:
    accuracy = 50*[0]
    training_instances = 50*[0]

    for limit in range(1,51):
        # load data
        knn = Knearest(data.train_x[:10*limit], data.train_y[:10*limit], 3)
        confusion = knn.confusion_matrix(data.test_x, data.test_y)
        print("Accuracy: %f" % knn.accuracy(confusion))
        accuracy[limit-1] = knn.accuracy(confusion)
        training_instances[limit-1] = 10*limit

    plt.plot(training_instances, accuracy)
    plt.xlabel('Training Instances')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Training Instances')

    plt.show()

# 1.3: which values get confused the most easily
if done2:
    b_accuracy = 3*[0]
    b_training_instances = 3*[0]

    for limit in range(1,4):
        knn = Knearest(data.train_x[:500*limit], data.train_y[:500*limit], 3)
        confusion = knn.confusion_matrix(data.test_x, data.test_y)
        b_accuracy[limit - 1] = knn.accuracy(confusion)
        b_training_instances[limit - 1] = limit

        print("Confusion matrix: maximum samples = " + str(500*limit))
        print("\t" + "\t".join(str(x) for x in range(10)))
        print("".join(["-"] * 90))
        for ii in range(10):
            print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                           for x in range(10)))
        print("Accuracy: %f" % knn.accuracy(confusion))

# 1.3: which values get confused the most easily
if done2b:
    knn = Knearest(data.train_x[:150], data.train_y[:150], 3)
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print(confusion)

# 1.4: what is the role of k on training accuracy
if done3:
    c_accuracy = [[0 for x in range(0,20)] for x in range(0,10)]
    c_training_instances = [[0 for x in range(0,20)] for x in range(0,10)]
    c_k_values = [[0 for x in range(0,20)] for x in range(0,10)]
    for limit in range(1,20):
        # load data
        for k in range(1,10):
            knn = Knearest(data.train_x[:10*limit], data.train_y[:10*limit], k)
            confusion = knn.confusion_matrix(data.test_x, data.test_y)
            c_accuracy[k][limit] = knn.accuracy(confusion)
            c_training_instances[k][limit] = 10*limit
            c_k_values[k][limit] = k
    c_accuracy = c_accuracy[1:]
    c_training_instances = c_training_instances[1:]
    c_k_values = c_k_values[1:]

    for ii in range(0,9):
        plt.plot(c_training_instances[:][0], c_accuracy[:][ii])
        break

    plt.xlabel('Training Instances')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Training Instances for different K values')

    plt.show()

# 1.5: in general does small k cause over-fitting or under-fitting?
if done4:
    d_accuracy = 7 * [0]
    d_training_instances = 7 * [0]

    for k in range(2, 9):
        knn = Knearest(data.train_x[:250], data.train_y[:250], k)
        confusion = knn.confusion_matrix(data.test_x, data.test_y)
        d_accuracy[k - 2] = knn.accuracy(confusion)
        d_training_instances[k - 2] = k
        # confusion matrix
        print("Confusion matrix: maximum samples = " + str(250))
        print("K  = " + str(k))
        print("\t" + "\t".join(str(x) for x in range(10)))
        print("".join(["-"] * 90))
        for ii in range(10):
            print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                           for x in range(10)))
        # accuracy
        print("Accuracy: %f" % knn.accuracy(confusion))


