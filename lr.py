#Goal : implement Natural Language Processing (NLP) system
#lr.py : implements a sentiment polarity analyzer using binary logistic regression
#predict a sentiment polarity for the corresponding feature vector of each movie review

import sys
import csv
from collections import OrderedDict
import math
import matplotlib.pyplot as plt

#store dictionary
def get_dict(dict_input):
    f = open(dict_input, 'r')
    input_f = csv.reader(f, delimiter =' ')
    dictionary = OrderedDict()
    for ith_row in input_f:
        dictionary[ith_row[0]] = ith_row[1]
    return dictionary

#store data
def store_data(input_file):
    f  = open(input_file, 'r')
    input_f = csv.reader(f, delimiter = '\t')
    X = []
    Y = []
    for ith_row in input_f:
        Y.append(int(ith_row[0]))
        x_i = OrderedDict()
        x_i[-1] = 1 #bias term added
        for i in range(1, len(ith_row)):
            key, value = ith_row[i].split(":")
            x_i[int(key)] = int(value)
        X.append(x_i)
    return [Y, X]

def sparse_dot(X, theta):
    product = 0.0
    for i,v in X.items():
        product += v * theta[i]
    return product

def single_SGD(Y_i, X_i, theta, learn_rate):
    val = sparse_dot(X_i, theta) #thetaT x_i
    for k,v in X_i.items():
        step = learn_rate * v * (Y_i - (math.exp(val) / (1 + math.exp(val))))
        theta[k] = theta[k] + step
    return theta


def neg_LL(Y, X, theta):
    tally = 0
    for i in range(len(Y)):
        val = sparse_dot(X[i], theta) #thetaT x_i
        tally += -Y[i] * val + math.log(1 + math.exp(val))
    return tally

#dictionary = W
def train(train_data, valid_data, dictionary ,num_epoch):
    Y = train_data[0]
    X = train_data[1]
    valid_Y = valid_data[0]
    valid_X = valid_data[1]
    train_avg_NLLs = []
    valid_avg_NLLs = []
    epochs = []
    learn_rate = 0.1
    #initialize
    theta = [0] * (len(dictionary) + 1) #bias term added
    for epoch in range(num_epoch): #epoch
        epochs.append(epoch)
        for i in range(len(Y)): #N
            theta = single_SGD(Y[i], X[i], theta, learn_rate)
        train_tally = neg_LL(Y, X, theta)
        valid_tally = neg_LL(valid_Y, valid_X, theta)
        train_avg = train_tally / len(Y)
        valid_avg = valid_tally / len(valid_Y)
        train_avg_NLLs.append(train_avg)
        valid_avg_NLLs.append(valid_avg)
    return (theta, train_avg_NLLs, valid_avg_NLLs, epochs)


def plot_analysis(train_avg_NLLs,valid_avg_NLLs , epochs):
    # plot analysis
    fig = plt.figure()
    #train
    plt.errorbar(epochs, train_avg_NLLs, label='Train Data')
    #validation
    plt.errorbar(epochs, valid_avg_NLLs, label='Validation Data')
    plt.legend(loc='lower right')
    plt.show()


#predict
def predict(data, theta, out_file):
    out = open(out_file, 'w')
    error = 0.0
    Y = data[0]
    X = data[1]
    for i in range(len(Y)):
        val = sparse_dot(X[i], theta)
        sigmoid = 1.0 / (1.0 + math.exp(-val))
        if sigmoid > 0.5:
            label = 1
        else:
            label = 0
        txt = str(label) + "\n"
        out.write(txt)
        if label != Y[i]:
            error += 1.0
    return error / len(Y)

if __name__ == '__main__':
    #path to the formatted .tsv file
    formatted_train_input = sys.argv[1]
    formatted_validation_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    #dictionary
    dict_input = sys.argv[4]
    #path to .labels file
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    #number of times SGD loops through all of the training data
    num_epoch = int(sys.argv[8])

    #get dictionary
    dictionary = get_dict(dict_input)

    #store data
    train_data = store_data(formatted_train_input)
    valid_data = store_data(formatted_validation_input)
    test_data = store_data(formatted_test_input)

    #train
    vals = train(train_data, valid_data, dictionary ,num_epoch)
    theta = vals[0]
    train_avg_NLLs = vals[1]
    valid_avg_NLLs = vals[2]
    epochs = vals[3]

    #validation
    plot_analysis(train_avg_NLLs,valid_avg_NLLs , epochs)

    #predict
    train_error = predict(train_data, theta, train_out)
    test_error = predict(test_data, theta, test_out)

    #metrics
    metrics_out_file = open(metrics_out, 'w')
    txts = "error (train) : " + str(train_error) + "\n" + "error (test) : " + str(test_error)
    print(txts)
    metrics_out_file.write(txts)
    metrics_out_file.close()



























