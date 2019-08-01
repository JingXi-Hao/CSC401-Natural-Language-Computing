from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp
# from tqdm import tqdm

dataDir = '/u/cs401/A3/data/'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))


def log_b_m_x( m, x, myTheta, preComputedForM=[] ):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

        NOTE: Assume the type of preComputedForM is always a list

    '''

    # define variables
    M, d = myTheta.mu.shape
    sigmaSquare = myTheta.Sigma[m] # ** 2, already squared

    # compute each term
    term1 = (0.5 * (x ** 2)) - (myTheta.mu[m] * x)
    term2 = np.divide(term1, sigmaSquare, out=np.zeros_like(sigmaSquare), where=(sigmaSquare != 0))

    # define result
    noPrecompute = (type(preComputedForM) != list) or (len(preComputedForM) != M)
    if noPrecompute: # once again, assume preComputedForM type is list
        # compute the rest of the terms
        # 1 x d --> scaler
        term3 = np.divide(myTheta.mu[m] ** 2, sigmaSquare, out=np.zeros_like(sigmaSquare), where=(sigmaSquare != 0))
        term3 = (0.5) * np.sum(term3)
        # scaler
        term4 = (d / float(2)) * (np.log(2 * np.pi))
        # 1 x d --> scaler
        term5 = (0.5) * np.sum(np.log(sigmaSquare, where=(sigmaSquare != 0))) # .reshape((M, 1))

        # define reutrn result and convert into list
        result = - np.sum(term2) - term3 - term4 - term5
        # print('compute')
    else:
        result = - np.sum(term2) - preComputedForM[m] # scaler
        # print('preocomputed used')

    # print ( 'log_b_m_x: {} \n'.format(result) )

    return result

    
def log_p_m_x( m, x, myTheta ):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''

    # # define variables 
    omegas = myTheta.omega
    M = omegas.shape[0]
    preComputedForM = preCompute(myTheta) # a list

    # use logsumexp to compute in a more stable way, compare with np.log(np.sum(np.exp())), 
    # seems to produce more precise output
    # also omegas should be in [0,1] check whether m-th omega is 0, if it is, then 
    # directly return zero
    if omegas[m, 0] <= 0: # should not be less than 0, but write this condition here
        return 0

    # compute log_p_m_x
    bmxs = []
    for i in range(0, M):
        bmxs.append(log_b_m_x( i, x, myTheta, preComputedForM ))

    bmxs = np.array(bmxs).reshape((M, 1))

    result = np.log(omegas[m, 0]) + log_b_m_x( m, x, myTheta, preComputedForM ) - logsumexp(bmxs, b=omegas)

    # print ( 'log_p_m_x: {} \n'.format(result) )

    return result
    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''

    # # take exp to remove log
    # result = np.sum(myTheta.omega * np.exp(log_Bs), axis=0) # sum over M --> 1 x T
    # result = np.sum(np.log(result, out=np.zeros_like(result), where=(result > 0))) # sum over T

    # use logsumexp to compute in a more stable way, also impossible that all omegas are zero since
    # they need to sum to 1
    result = np.sum(logsumexp(log_Bs, b=myTheta.omega, axis=0))
    
    # print( 'logLik: {} \n'.format(result) )

    return result

    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    myTheta = theta( speaker, M, X.shape[1] )
    #print ('TODO')

    # define variables
    T, d = X.shape

    # initialize
    ind = random.sample(range(T), M)
    myTheta.mu = X[np.array(ind)]
    myTheta.Sigma = np.ones((M, d)) # this Mxd matrix consists of M diagonals of dxd matrix
    myTheta.omega[..., 0] = float(1) / M
    i = 0
    prev_L = float('-inf')
    improvement = float('inf')
    # log_Bs = np.zeros((M, T))
    # log_Ps = np.zeros((M, T))

    while i <= maxIter and improvement >= epsilon:
        preComputedForM = np.array(preCompute(myTheta)).reshape((M, 1)) # M x 1

        # # compute log_Bs
        # # nested loop --- really slow for training
        # for m in tqdm(range(0, M)):
        #     for t in tqdm(range(0, T)):
        #         # log_Bs[m, t] = log_b_m_x( m, X[t], myTheta )
        #         log_Ps[m, t] = log_p_m_x( m, X[t], myTheta )
        # print("for loop: {}".format(log_Ps))

        # for efficiency, use matrix operation to compute log_Bs
        sigmaSquare = np.reciprocal(myTheta.Sigma, where=(myTheta.Sigma != 0)) # M x d
        xSquare = (0.5 * (X ** 2)).T # d x T
        term1 = (-1) * np.dot(sigmaSquare, xSquare) # M x T
        term2 = np.multiply(myTheta.mu, sigmaSquare) # M x d
        term3 = np.dot(term2, X.T) # M x T
        log_Bs = term1 + term3 - preComputedForM
        # print(log_Bs)

        # compute likelihood and update loop constraints
        L = logLik( log_Bs, myTheta )
        improvement = L - prev_L
        prev_L = L
        i += 1

        # compute Ps for the purpose of updating parameters
        # term4 = myTheta.omega * np.exp(log_Bs) # M x T
        # term5 = np.sum(term4, axis=0) # 1 x T
        # Ps = np.divide(term4, term5, out=np.zeros_like(term4), where=(term5 > 0)) # M x T

        # use logsumexp to compute in a more stable way
        term4 = np.log(myTheta.omega) + log_Bs - logsumexp(log_Bs, b=myTheta.omega, axis=0)
        Ps = np.exp(term4) # make sure Ps >= 0
        # print(term4)
        # print(Ps)

        # update parameters
        term6 = np.sum(Ps, axis=1).reshape((M, 1))
        myTheta.omega = term6 / float(T) # M times 1
        term7 = np.dot(Ps, X)
        myTheta.mu = np.divide(term7, term6, out=np.zeros_like(term7), where=(term6 != 0)) # M times d and M times 1 --> M x d
        term8 = np.dot(Ps,  X ** 2)
        myTheta.Sigma = np.divide(term8, term6, out=np.zeros_like(term8), where=(term6 != 0)) - (myTheta.mu ** 2) # M x d
        # print(myTheta.Sigma)

    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    # print ('TODO')

    # # open file for write the test information
    # if os.path.isfile('./gmmLiks.txt'):
    #     outputFile = open("gmmLiks.txt", 'a')
    # else:
    #     outputFile = open("gmmLiks.txt", 'w')
    #     outputFile.write('Test results with randomSeed = 0 and k = 5 \n\n')

    # define variables
    info = []
    T, d = mfcc.shape
    M = models[0].mu.shape[0]
    length = len(models)

    # loop over each model and compute log likelihood
    for i in range(0, length):

        myTheta = models[i]
        preComputedForM = np.array(preCompute(myTheta)).reshape((M, 1))
        # log_Bs = np.zeros((M, T))

        # for efficiency, use matrix operation to compute log_Bs
        sigmaSquare = np.reciprocal(myTheta.Sigma, where=(myTheta.Sigma != 0)) # M x d
        xSquare = (mfcc ** 2).T # d x T
        term1 = (-0.5) * np.dot(sigmaSquare, xSquare) # M x T
        term2 = np.multiply(myTheta.mu, sigmaSquare) # M x d
        term3 = np.dot(term2, mfcc.T) # M x T
        log_Bs = term1 + term3 - preComputedForM

        # compute for likelihood
        L = logLik(log_Bs, myTheta)
        info.append((i, myTheta, L))

    # check whether model info is valid (model list is empty or not)
    if len(info) > 0:
        # sort model based on log likelihood and update the best model index
        info = sorted(info, key=lambda x: x[2])
        bestModel = info[-1][0] # get model index

        # print as requested
        if (k is not None) and (k > 0):
            print("{}".format(models[correctID].name))
            # outputFile.write("{} \n".format(models[correctID].name))

            k = min(k, length) # in case that k is larger than length

            for j in range(length - 1, length - k - 1, -1):
                print("{} {}".format(info[j][1].name, info[j][2]))
                # outputFile.write("{} {} \n".format(info[j][1].name, info[j][2]))
            # outputFile.write('\n\n')
            # outputFile.close()

    return 1 if (bestModel == correctID) else 0

# helper function to pre-compute for M (terms that are not depend on X)
def preCompute(myTheta):
    # define variables
    sigmaSquare = myTheta.Sigma # ** 2, no need to square, already squared
    M, d = myTheta.mu.shape

    # compute each term
    # M x 1 --> (M,)
    term1 = np.divide(myTheta.mu ** 2, sigmaSquare, out=np.zeros_like(sigmaSquare), where=(sigmaSquare != 0))
    term1 = (0.5) * np.sum(term1, axis=1) # .reshape((M, 1))
    # scaler
    term2 = (d / float(2)) * (np.log(2 * np.pi))
    # M x 1 --> (M,)
    term3 = (0.5) * np.sum(np.log(sigmaSquare, where=(sigmaSquare != 0)), axis=1) # .reshape((M, 1))

    # define reutrn result and convert into list
    result = (term1 + term2 + term3).flatten()

    return result.tolist()


if __name__ == "__main__":
    # set seed here
    random.seed(0) # make sure to produce the same result every time

    trainThetas = []
    testMFCCs = []
    # print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)
    print("Accuracy: {0: 1.4f} \n".format(accuracy))

    # # write into file for Section 2.3
    # if os.path.isfile('./gmmLiks.txt'):
    #     outputFile = open("gmmLiks.txt", 'a')
    # else:
    #     outputFile = open("gmmLiks.txt", 'w')
    # outputFile.write("Accuracy: {0: 1.4f} \n".format(accuracy))
    # outputFile.close()

    # ########################## code below is for experiments purpose #########################
    # # open file to write
    # if os.path.isfile('./gmmDiscussion.txt'):
    #     outputFile = open("gmmDiscussion.txt", 'a')
    # else:
    #     outputFile = open("gmmDiscussion.txt", 'w')

    # # define variables
    # d = 13
    # k = 5  # number of top speakers to display, <= 0 if none
    # MList = [1, 2, 3, 5, 7, 8, 10, 12, 18, 25]
    # epsilon = 0.0
    # maxIterList = [0, 1, 3, 5, 7, 10, 15, 20, 30, 50]
    # speakersList = [1, 5, 8, 15, 20, 25, 32]
    # random.seed(0)

    # # original value for M, maxIter, and speakers (32)
    # # M = 8
    # maxIter = 20
    # numSpeakers = 32

    # ################################## test M/maxIter/speakers ##################################
    # outputFile.write('Test M with maxIter = 20 and Speakers = 32 \n\n')
    # # outputFile.write('Test maxIter with M = 8 and Speakers = 32 \n\n')
    # # outputFile.write('Test Speakers with M = 8 and maxIter = 20 \n\n')
    # # outputFile.write('Test Speakers with M = 8 and maxIter = 20 and totalSpeakers = 32 \n\n')
    # for M in tqdm(MList):
    #     trainThetas = []
    #     testMFCCs = []
    #     count = 0
    #     for subdir, dirs, files in os.walk(dataDir):
    #         for speaker in dirs:
    #             # print( speaker )

    #             files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
    #             random.shuffle( files )
                
    #             # if count < numSpeakers:
    #             testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
    #             testMFCCs.append( testMFCC )
    #             # if count < numSpeakers:
    #             X = np.empty((0,d))
    #             for file in files:
    #                 myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
    #                 X = np.append( X, myMFCC, axis=0)

    #             trainThetas.append( train(speaker, X, M, epsilon, maxIter) )
    #             # count += 1
    #             # else:
    #                 # break
    #                 # trainThetas.append( theta( speaker, M, d ) )

    #     # evaluate 
    #     numCorrect = 0
    #     for i in range(0,len(testMFCCs)):
    #         numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    #     accuracy = 1.0*numCorrect/len(testMFCCs)
    #     outputFile.write("M: {0} \t Accuracy: {1: 1.4f} \n".format(M, accuracy))
    #     # outputFile.write("maxIter: {0} \t Accuracy: {1: 1.4f} \n".format(maxIter, accuracy))
    #     # outputFile.write("Total Speakers: {0} \t Accuracy: {1: 1.4f} \n".format(numSpeakers, accuracy))
    #     # outputFile.write("Known Training Speakers: {0} \t Accuracy: {1: 1.4f} \n".format(numSpeakers, accuracy))
    # outputFile.write('\n\n')
    # outputFile.close()
