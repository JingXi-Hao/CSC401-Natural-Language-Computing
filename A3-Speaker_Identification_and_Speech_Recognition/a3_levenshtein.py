import os
import numpy as np
import re
import string
from scipy import stats
# from tqdm import tqdm

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """

    # define variables
    N = len(r)
    M = len(h)

    # deal with special cases
    if N == 0 and M == 0:
        return (float(0), 0, 0, 0) # if both empty string, then no error
    # these two cases are automatically dealt, define here to reduce backtrack
    if N == 0 and M != 0:
        return (np.inf, 0, M, 0) # np.inf due to M / 0
    if N != 0 and M == 0:
        return (1.0, 0, 0, N) # 1.0 due to N / N

    # define variables
    R = np.zeros((N+1, M+1)) # matrix of distances
    B = np.zeros((N+1, M+1), dtype=object) # backtracking matrix

    # set infinity when i or j equals to 0, except R[0, 0] = 0
    # R[0, :] = np.inf
    # R[:, 0] = np.inf
    # R[0, 0] = 0
    R[0, :] = np.array(list(range(M+1))) # 0, 1, 2, 3...
    R[:, 0] = np.array(list(range(N+1))) # 0, 1, 2, 3...
    B[0, :] = 'left' # first row (0) is insertion errors
    B[:, 0] = 'up' # first column (0) is deletion errors
    B[0,0] = '' # (0,0) no need to considered

    # update R and B
    for i in range(1, N+1):
        for j in range(1, M+1):
            # define past errors + 1 deletion error
            del_error = R[i-1, j] + 1

            # define past errors + 1 substitution error
            sub_error = R[i-1, j-1]
            if r[i-1] != h[j-1]:
                sub_error += 1

            # define past errors + 1 insertion error
            ins_error = R[i, j-1] + 1

            # find the minimum sum
            R[i, j] = min([del_error, sub_error, ins_error])

            # update B matrix
            if R[i, j] == del_error:
                B[i, j] = 'up'
            elif R[i, j] == ins_error:
                B[i, j] = 'left'
            else:
                B[i, j] = 'up-left'

    # define number of deletions, insertions, and substitutions from the path
    counts = [0, 0] # nI, nD, where nS is equal to sum - nI - nD
    t = N
    s = M
    while True:
        # keep going until reach (0,0), once reach then stop
        if t <= 0 and s <= 0:
            break
        # define step and make sure it is valid
        step = B[t, s]
        # count number of deletions, insertions, and substitutions
        if step == 'up':
            counts[1] += 1
            t = t - 1
        if step == 'left':
            counts[0] += 1
            s = s - 1
        if step == 'up-left': # up-left
            # counts[0] += 1 # not always plus one, since maybe plus zero, so use total minus nI and nD
            t = t - 1
            s = s - 1

    # define error-rate
    error = R[N, M] / float(N)

    # define result
    nS = np.rint(R[N, M] - counts[0] - counts[1]).astype(int)
    result = (error, nS, counts[0], counts[1])

    return result

# helper function to preprocess each line
def preprocess(line):
    output = ''
    lineCopy = line

    # step 1: remove newline character, convert into lower case letters and add sentence tags for it
    line = line.replace("\n", " ").replace("\r", " ")
    line = line.lower()
    line = re.sub(r"\s+", " ", line) 
    line = line.strip()

    # remove all punctuations except []
    line = re.sub(r"([\\!\"#$%&\()*+,-./:;<=>?@^_`{|}~\']+)", " ", line)
    line = re.sub(r"\s+", " ", line) 
    line = line.strip()

    # deal with '[word]' case
    pattern = re.compile(r"(\[)(?![\w]+\])")
    line = re.sub(pattern, " ", line)
    pattern = re.compile(r"(?<![\w])(\])")
    line = re.sub(pattern, " ", line)
    def removeBrackets(match):
        if match.group(1)[0] != "[":
            return " " + match.group(1) + " "
        else:
            return match.group(0)
    pattern = re.compile(r"([\S]+)(\])")
    line = re.sub(pattern, removeBrackets, line)
    line = re.sub(r"\s+", " ", line) 
    line = line.strip()

    output = line
    return output

if __name__ == "__main__":
    # print( 'TODO' )

    # open file to wirte
    outputFile = open("asrDiscussion.txt", 'w')

    # define variables
    googleErrors = []
    kaldiErrors = []

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )
            basePath = os.path.join(dataDir, speaker)
            refPath = os.path.join(basePath, 'transcripts.txt')
            googlePath = os.path.join(basePath, 'transcripts.Google.txt')
            kaldiPath = os.path.join(basePath, 'transcripts.Kaldi.txt')

            # read files
            refFile = open(refPath, 'r')
            refLines = refFile.read().split('\n')
            if refLines[-1] == '':
                refLines = refLines[0:-1]
            refFile.close()

            googleFile = open(googlePath, 'r')
            googleLines = googleFile.read().split('\n')
            if googleLines[-1] == '':
                googleLines = googleLines[0:-1]
            googleFile.close()

            kaldiFile = open(kaldiPath, 'r')
            kaldiLines = kaldiFile.read().split('\n')
            if kaldiLines[-1] == '':
                kaldiLines = kaldiLines[0:-1]
            kaldiFile.close()

            # determine length
            length = min(len(refLines), len(googleLines), len(kaldiLines))

            # loop over each line
            for i in range(0, length):
                ref = preprocess(refLines[i]).split()
                google = preprocess(googleLines[i]).split()
                kaldi = preprocess(kaldiLines[i]).split()

                # compute error rate
                googleResult = Levenshtein(ref, google)
                kaldiResult = Levenshtein(ref, kaldi)
                googleErrors.append(googleResult[0])
                kaldiErrors.append(kaldiResult[0])

                # write computed results into file
                outputFile.write('{0} {1} {2} {3: 1.4f} S:{4}, I:{5}, D:{6} \n'.format(speaker, 'Google', i, 
                    googleResult[0], googleResult[1], googleResult[2], googleResult[3]))

                outputFile.write('{0} {1} {2} {3: 1.4f} S:{4}, I:{5}, D:{6} \n'.format(speaker, 'Kaldi', i, 
                    kaldiResult[0], kaldiResult[1], kaldiResult[2], kaldiResult[3]))
            outputFile.write('\n\n')

    # convert into numpy array
    googleErrors = np.array(googleErrors)
    kaldiErrors = np.array(kaldiErrors)

    # compute mean and standard deviation, define t-test, and write into file
    t_value, p_value = stats.ttest_ind(googleErrors, kaldiErrors, equal_var = False)
    outputFile.write('Google WER Average: {0: 1.4f}, Google WER Standard Deviation: {1: 1.4f}, \
        Kaldi WER Average: {2: 1.4f}, Kaldi WER Standard Deviation: {3: 1.4f}, \
        Calculate T-test for Google WER and Kaldi WER:  T-value: {4: 1.4f}  P-value: {5} \n'.format(
        np.mean(googleErrors), np.std(googleErrors), np.mean(kaldiErrors), np.std(kaldiErrors), t_value, p_value))

    # close the file
    outputFile.close()
