from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import argparse
import sys
import os

# add new imports
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import csv
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold
from scipy import stats

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    acc = 0
    if np.sum(C) != 0:
        acc = np.divide(np.trace(C), np.sum(C))
    return acc

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    den = np.sum(C, axis=1)
    rec = np.divide(np.diagonal(C), den)
    return rec

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    den = np.sum(C, axis=0)
    pre = np.divide(np.diagonal(C), den)
    return pre
    

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    print('start class31\n')
    # load and split the data into 80% for training and 20% for testing
    data = np.load(filename)['arr_0']
    #print(np.argwhere(np.isnan(data)))
    X = data[:, 0:173]
    y = data[:, 173]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("split data is done!\n")

    # set iBest default to be 0 and open csv file for writing
    iBest = 0
    maxAcc = float('-inf')
    csvfile = open('a1_3.1.csv', 'w')
    writer = csv.writer(csvfile, delimiter=',')

    # define classifiers
    clf1 = LinearSVC(loss="hinge")
    clf2 = SVC(kernel="rbf", gamma=2, max_iter=1000)
    clf3 = RandomForestClassifier(n_estimators=10, max_depth=5)
    clf4 = MLPClassifier(alpha=0.05)
    clf5 = AdaBoostClassifier()
    clfList = [clf1, clf2, clf3, clf4, clf5]

    clfNum = 0
    # train each classifier
    for clf in clfList:
        clfNum += 1
        print("clf {} training starts! \n".format(clfNum))

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        con_matrix = confusion_matrix(y_test, y_pred)

        print("clf {} is ready! \n".format(clfNum))

        # compute accuracy, recall, precision
        acc = accuracy(con_matrix)
        rec = recall(con_matrix)
        pre = precision(con_matrix)

        print("clf {} acc : {} \n".format(clfNum, acc))

        # build list of information for csv file
        valList = np.append(clfNum, acc)
        valList = np.append(valList, rec)
        valList = np.append(valList, pre)
        valList = np.append(valList, con_matrix.reshape((16,)))

        # write measurements information into file
        writer.writerow(valList)

        print("clf {} info. list : {} \n".format(clfNum, valList))

        if maxAcc < acc:
            maxAcc = acc
            iBest = clfNum

    print("iBest: {} \n".format(iBest))

    # close the csv file opened
    csvfile.close()   

    return (X_train, X_test, y_train, y_test,iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
    '''
    print("start class32 \n")

    sampleSize = [1000, 5000, 10000, 15000, 20000]
    accList = []

    for size in sampleSize:
        print("sample size: {} \n".format(size))

        # every time change the sample size, redefine the classifier
        if iBest == 1:
            clf = LinearSVC(loss="hinge")
        if iBest == 2:
            clf = SVC(kernel="rbf", gamma=2, max_iter=1000)
        if iBest == 3:
            clf = RandomForestClassifier(n_estimators=10, max_depth=5)
        if iBest == 4:
            clf = MLPClassifier(alpha=0.05)
        if iBest == 5:
            clf = AdaBoostClassifier()

        # randomly select "size" number of samples
        rows = X_train.shape[0]
        idxList = np.random.randint(rows, size=size)
        X_selected = X_train[idxList, :]
        y_selected = y_train[idxList]

        # train classifier
        print("fit classifier for size: {} \n".format(size))

        clf.fit(X_selected, y_selected)
        y_pred = clf.predict(X_test)
        con_matrix = confusion_matrix(y_test, y_pred)
        acc = accuracy(con_matrix)
        accList.append(acc)

        # set X_1k and y_1k
        if size == 1000:
            X_1k = X_selected
            y_1k = y_selected

    print("acclist for class32: {}\n".format(accList))

    # write accuracies into a csv file
    with open('a1_3.2.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(accList)
    csvfile.close()

    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('start class33 \n')

    kList = [5, 10, 20, 30, 40, 50]
    accList = []
    features_32 = []
    features_1 = []
    p_1 = []

    # copy selector for top 5 feature case
    selector_5_32 = None
    selector_5_1 = None
    X_5_32 = None
    X_5_1 = None

    # open csv file for writing
    csvfile = open('a1_3.3.csv', 'w')
    writer = csv.writer(csvfile, delimiter=',')

    for k in kList:
        # select top k features for 32K training data
        print("select features for 32K with k = {}\n".format(k))

        selector_32 = SelectKBest(f_classif, k)
        X_new_32 = selector_32.fit_transform(X_train, y_train)
        pp_32 = selector_32.pvalues_
        top_f_32 = np.sort(pp_32, axis=None)[0:k]
        features_32.append(selector_32.get_support(indices=True))

        # define top feature p values list
        print("find top {} features p values for 32K \n".format(k))
        p_list = np.append(k, top_f_32)
        writer.writerow(p_list)

        print("results for writing csv file: \n {} \n".format(p_list))

        # select top k features for 1K training data
        print("select features for 1K with k = {}\n".format(k))

        selector_1 = SelectKBest(f_classif, k)
        X_new_1 = selector_1.fit_transform(X_1k, y_1k)
        pp_1 = selector_1.pvalues_
        top_f_1 = np.sort(pp_1, axis=None)[0:k]
        features_1.append(selector_1.get_support(indices=True))
        p_1.append(np.append(k, top_f_1))

        # for k equals to 5 case, save the transformed X and trained selector for later use
        if k == 5:
            selector_5_32 = selector_32
            selector_5_1 = selector_1
            X_5_32 = X_new_32
            X_5_1 = X_new_1

    # deal with top 5 feature case for both 32K and 1K
    # define the best classifier for each sample size
    if i == 1:
        clf_32 = LinearSVC(loss="hinge")
        clf_1 = LinearSVC(loss="hinge")
    if i == 2:
        clf_32 = SVC(kernel="rbf", gamma=2, max_iter=1000)
        clf_1 = SVC(kernel="rbf", gamma=2, max_iter=1000)
    if i == 3:
        clf_32 = RandomForestClassifier(n_estimators=10, max_depth=5)
        clf_1 = RandomForestClassifier(n_estimators=10, max_depth=5)
    if i == 4:
        clf_32 = MLPClassifier(alpha=0.05)
        clf_1 = MLPClassifier(alpha=0.05)
    if i == 5:
        clf_32 = AdaBoostClassifier()
        clf_1 = AdaBoostClassifier()

    print("use 5 top features to fit the 32K clf. \n")

    # train classifier using top 5 features for 32K
    X_32_test_with_top_f = selector_5_32.transform(X_test)

    clf_32.fit(X_5_32, y_train)
    y_pred_32 = clf_32.predict(X_32_test_with_top_f)
    con_matrix_32 = confusion_matrix(y_test, y_pred_32)
    acc_32 = accuracy(con_matrix_32)

    print("use 5 top features to fit the 1K clf. \n")

    # train classifier using top 5 features for 1K
    X_1_test_with_top_f = selector_5_1.transform(X_test)

    clf_1.fit(X_5_1, y_1k)
    y_pred_1 = clf_1.predict(X_1_test_with_top_f)
    con_matrix_1 = confusion_matrix(y_test, y_pred_1)
    acc_1 = accuracy(con_matrix_1)

    # build accuracy information for writing in csvfile
    accList.append(acc_1)
    accList.append(acc_32)
    writer.writerow(accList)

    # close csv file
    csvfile.close()

    # # features file for 32K
    # with open('features_32.csv', 'w') as csvfile1:
    #     writer = csv.writer(csvfile1, delimiter=',')
    #     for j in range(0, len(features_32)):
    #         writer.writerow(features_32[j])
    # csvfile1.close()

    # # features file for 1K
    # with open('features_1.csv', 'w') as csvfile2:
    #     writer = csv.writer(csvfile2, delimiter=',')
    #     for j in range(0, len(features_1)):
    #         writer.writerow(features_1[j])
    # csvfile2.close()

    # # p values for 1K
    # with open('pvals_1.csv', 'w') as csvfile3:
    #     writer = csv.writer(csvfile3, delimiter=',')
    #     for j in range(0, len(p_1)):
    #         writer.writerow(p_1[j])
    # csvfile3.close()


def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
    '''
    print(" start class34 \n")

    data = np.load(filename)['arr_0']
    X = data[:, 0:173]
    y = data[:, 173]

    # define kfold
    kf = KFold(n_splits=5, shuffle=True)
    fold = 0

    # define outputs
    results = []
    p_vals = []

    # open csv file
    csvfile = open('a1_3.4.csv', 'w')
    writer = csv.writer(csvfile, delimiter=',')

    for train_idx, test_idx in kf.split(X):
        fold += 1
        print(" {} fold is running! \n".format(fold))

        # define training and test set
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_test = X[test_idx, :]
        y_test = y[test_idx]

        # define classifiers
        clf1 = LinearSVC(loss="hinge")
        clf2 = SVC(kernel="rbf", gamma=2, max_iter=1000)
        clf3 = RandomForestClassifier(n_estimators=10, max_depth=5)
        clf4 = MLPClassifier(alpha=0.05)
        clf5 = AdaBoostClassifier()
        clfList = [clf1, clf2, clf3, clf4, clf5]

        # compute accuracy for each classifiers at each fold
        accList = []
        for clf in clfList:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            con_matrix = confusion_matrix(y_test, y_pred)
            acc = accuracy(con_matrix)
            accList.append(acc)
        results.append(accList)

        # write into csv file
        writer.writerow(accList)

    print("results (accList) for class34: {} \n".format(results))

    # convert results from list to numpy array and compute p-values for t-test
    results = np.array(results)
    for index in range(0, results.shape[1]):
        if index != (i-1):
            best = results[:, i-1]
            S = stats.ttest_rel(best, results[:, index])
            p_vals.append(S.pvalue)

    print("p_vals for class34: {} \n".format(p_vals))

    # write into csv file
    writer.writerow(p_vals)
    csvfile.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
    X_train, X_test, y_train, y_test,iBest = class31(args.input)

    print(" ---------------------------------------------------------------------------------------- \n\n")

    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)

    print(" ---------------------------------------------------------------------------------------- \n\n")

    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)

    print(" ---------------------------------------------------------------------------------------- \n\n")

    class34(args.input, iBest)
