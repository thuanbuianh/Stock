import os
import json
import math
import torch
import numpy as np
import pandas as pd
import argparse
import timeit
import TimeSeries

import wrappers


def load_UEA_dataset(path, dataset):
    """
    Loads the UEA dataset given in input in numpy arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    trainTS = {}
    testTS = {}
    
    train_raw = pd.read_csv((path + "/" + dataset + "/" + dataset + "_TRAIN.txt"), delim_whitespace=True, header=None)
    test_raw = pd.read_csv((path + "/" + dataset + "/" + dataset + "_TEST.txt"), delim_whitespace=True, header=None)

    for i in range(train_raw.shape[0]):
        label = int(train_raw.iloc[i, 0])
        series = train_raw.iloc[i,1:].tolist()
        trainTS[i] = TimeSeries(series, label)
        trainTS[i].NORM(True)

    for i in range(test_raw.shape[0]):
        label = int(test_raw.iloc[i, 0])
        series = test_raw.iloc[i, 1:].tolist()
        testTS[i] = TimeSeries(series, label)
        testTS[i].NORM(True)

    train = np.array([np.array(trainTS[i].data) for i in range(len(trainTS))])
    train = train.reshape(train.shape[0], 1, train.shape[1])
    train_labels = np.array([np.array(trainTS[i].label) for i in range(len(trainTS))])
    
    test = np.array([np.array(testTS[i].data) for i in range(len(testTS))])
    test = test.reshape(test.shape[0], 1, test.shape[1])
    test_labels = np.array([np.array(testTS[i].label) for i in range(len(testTS))])
        
    print('dataset load succeed !!!')
    return train, train_labels, test, test_labels


def fit_parameters(file, train, train_labels, test, test_labels, cuda, gpu, save_path, cluster_num,
                        save_memory=False):
    """
    Creates a classifier from the given set of parameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    classifier = wrappers.CausalCNNEncoderClassifier()

    # Loads a given set of parameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    params['in_channels'] = 1
    params['cuda'] = cuda
    params['gpu'] = gpu
    classifier.set_params(**params)
    return classifier.fit(
        train, train_labels, test, test_labels, save_path, cluster_num, save_memory=save_memory, verbose=True
    )

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UEA repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the dataset is located')
    parser.add_argument('--save_path', type=str, metavar='PATH', required=True,
                        help='path where the estimator is/should be saved')
    parser.add_argument('--cuda', action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE', required=True,
                        help='path of the file of parameters to use ' +
                             'for training; must be a JSON file')
    parser.add_argument('--load', action='store_true', default=False,
                        help='activate to load the estimator instead of ' +
                             'training it')
    parser.add_argument('--fit_classifier', action='store_true', default=False,
                        help='if not supervised, activate to load the ' +
                             'model and retrain the classifier')

    print('parse arguments succeed !!!')
    return parser.parse_args()


if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_arguments()
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False

    train, train_labels, test, test_labels = load_UEA_dataset(
        args.path, args.dataset
    )
    cluster_num = 2
    if not args.load and not args.fit_classifier:
        print('start new network training')
        classifier = fit_parameters(
            args.hyper, train, train_labels, test, test_labels, args.cuda, args.gpu, args.save_path, cluster_num,
            save_memory=False
        )
    else:
        classifier = wrappers.CausalCNNEncoderClassifier()
        hf = open(
            os.path.join(
                args.save_path, args.dataset + '_parameters.json'
            ), 'r'
        )
        hp_dict = json.load(hf)
        hf.close()
        hp_dict['cuda'] = args.cuda
        hp_dict['gpu'] = args.gpu
        classifier.set_params(**hp_dict)
        classifier.load(os.path.join(args.save_path, args.dataset))

    if not args.load:
        if args.fit_classifier:
            classifier.fit_classifier(classifier.encode(train), train_labels)
        classifier.save(
            os.path.join(args.save_path, args.dataset)
        )
        with open(
            os.path.join(
                args.save_path, args.dataset + '_parameters.json'
            ), 'w'
        ) as fp:
            json.dump(classifier.get_params(), fp)

    end = timeit.default_timer()
    print("All time: ", (end- start)/60)
