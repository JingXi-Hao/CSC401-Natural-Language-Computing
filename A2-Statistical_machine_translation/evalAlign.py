#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle

import decode
from align_ibm1 import *
from BLEU_score import *
from lm_train import *
from preprocess import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

By testing with distinct values of max iterations for EM step, the BLEU scores starts to become stable
when the max iterations equals to 10. I tried with 2, 5, 10, 20, 50 and notice that the BLEU scores 
have much differences as max_iter equals to 2 , 5, and 10 while it becomes quite stable as max_iter
equals to 10, 20, and 50. Therefore, I choose 10 as the value for max_iter.

Based on the way to build the AM model, we only consider one-to-one word alignment and ignore 
the semantics and grammar of the language. Thus, when translating french sentence into English, 
the words are translated into the french order and not follows the English grammar and semantics.
Therefore, the bigram and trigram BLEU scores are not as good as the unigram produces for any training 
size, where higher BLEU score represents a better match between references and candidate. This makes 
sense since as the token gets longer (n gets bigger), the order of the words (semantics and grammar) 
becomes more important and the probability of the occurrence of the token in the references would be 
less. Therefore, the BLEU score would be smaller and even could be zero which means that the candidate 
and the references do not match at all. Moreover, for the unigram case, based on the results, the BLEU 
scores gets better as the number of training data increases in general, which makes sense since the 
probability of having “correct” alignment would be higher as more training data given. Also, for each
the training size, the BLEU score decreases as n increases and the reason is explained before.

In comparison of the references, some reference would be more biased than others and having more 
references may not produce better BLEU scores. The quality of the references are essential. If 
we produce more references with high quality, then our BLEU score would get higher and more matches 
would be found. If more random references are given, then the BLEU would have a chance to get worse. 
Hence, the quality is much more essential than the quantity.

"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model
    """
    if use_cached == True:
        with open(fn_LM+'.pickle', 'rb') as handle:
            LM = pickle.load(handle)
    else:
        LM = lm_train(data_dir, language, fn_LM)
    return LM

def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data 
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model 
    """
    if use_cached == True:
        with open(fn_AM+'.pickle', 'rb') as handle:
            AM = pickle.load(handle)
    else:
        AM = align_ibm1(data_dir, num_sent, max_iter, fn_AM)
    return AM

def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """
    scores = []

    for i in range(0, len(eng_decoded)):
        candidate = eng_decoded[i]
        references = [eng[i], google_refs[i]]
        precision = 1

        # if n == 1:
        #     print("candidate {}: {} \n".format(i, candidate))
        #     print("ref_eng {}: {} \n".format(i, eng[i]))
        #     print("ref_google {}: {} \n".format(i, google_refs[i]))

        if n == 1:
            bleu_score = BLEU_score(candidate, references, n, True)
            scores.append(bleu_score)
        else:
            for j in range(1, n+1):
                precision *= BLEU_score(candidate, references, j, False)
            precision_term= precision ** (1/n)
            
            # compute brevity
            len_ref_e = len(eng[i].split())
            len_ref_g = len(google_refs[i].split())
            len_can = len(candidate.split())
            if abs(len_ref_e - len_can) < abs(len_ref_g - len_can):
                best_match = len_ref_e
            else:
                best_match = len_ref_g
            brevity = best_match / float(len_can)
            if brevity < 1:
                BP = 1
            else:
                BP = math.exp(1 - brevity)

            # compute bleu_score
            bleu_score = BP * precision_term
            scores.append(bleu_score)

    return scores


def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """


    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

    '''
    f = open("Task5.txt", 'w+')
    f.write(discussion) 
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    for i, AM in enumerate(AMs):
        
        f.write(f"\n### Evaluating AM model: {AM_names[i]} ### \n")
        # Decode using AM #
        # Eval using 3 N-gram models #
        all_evals = []
        for n in range(1, 4):
            f.write(f"\nBLEU scores with N-gram (n) = {n}: ")
            evals = _get_BLEU_scores(...)
            for v in evals:
                f.write(f"\t{v:1.4f}")
            all_evals.append(evals)

        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()
    '''
    # define variables
    trainging_sizes = [1000, 10000, 15000, 30000]
    max_iter = 10 # tried with 2, 5, 10, 20, 50
    n_sizes = [1, 2, 3]
    all_evals = []

    # define directories
    base_dir = '/u/cs401/A2_SMT/data/Hansard/'
    train_dir = base_dir + 'Training/'
    test_dir  = base_dir + 'Testing/'
    test_french = test_dir + 'Task5.f'
    reference_english = test_dir + 'Task5.e'
    reference_google_english = test_dir + 'Task5.google.e'
    fn_LME = 'LM_E'
    # fn_LMF = 'LM_F' # no need for this

    # compute LM model for english
    LM_E = _getLM(train_dir, 'e', fn_LME, False)
    print("\n### Train LM model is done ### \n")
    # LM_F = lm_train(train_dir, 'f', fn_LMF) # no need for this

    # compute 4 AM models
    AM_models = []
    for size in trainging_sizes:
        fn_AM = 'am_{}'.format(size)
        AM = _getAM(train_dir, size, max_iter, fn_AM, False)
        print("\n### Train AM model: {} is done ### \n".format(size))
        AM_models.append(AM)

    # get lines
    test_file = open(test_french, 'r')
    test_lines = test_file.read().split('\n')
    if '' in test_lines:
        test_lines = test_lines[0:-1]
    test_file.close()

    ref_eng_file = open(reference_english, 'r')
    ref_eng_lines = ref_eng_file.read().split('\n')
    if '' in ref_eng_lines:
        ref_eng_lines = ref_eng_lines[0:-1]
    ref_eng_file.close()

    ref_google_file = open(reference_google_english, 'r')
    ref_google_lines = ref_google_file.read().split('\n')
    if '' in ref_google_lines:
        ref_google_lines = ref_google_lines[0:-1]
    ref_google_file.close()

    # open file to wirte
    output_file = open("Task5.txt", 'w+')
    output_file.write(discussion) 
    output_file.write("\n\n")
    output_file.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    # preprocess english refs and google refs
    ref_e_pre = []
    ref_g_pre = []
    for k in range(0, len(test_lines)):
        ref_e = preprocess(ref_eng_lines[k], 'e')
        ref_g = preprocess(ref_google_lines[k], 'e')
        ref_e_pre.append(ref_e)
        ref_g_pre.append(ref_g)

    for i, AM in enumerate(AM_models):
        output_file.write("\n### Evaluating AM model: {} ### \n".format(trainging_sizes[i]))
        print("\n### Evaluating AM model: {} ### \n".format(trainging_sizes[i]))

        lines_decoded = []
        for j in range(0, len(test_lines)):
            french_line = preprocess(test_lines[j], 'f')
            #print(french_line)
            line_for_test = decode.decode(french_line, LM_E, AM)
            lines_decoded.append(line_for_test)

        for n in n_sizes:
            output_file.write("\nBLEU scores with N-gram (n) = {}: ".format(n))
            print("\nBLEU scores with N-gram (n) = {}: ".format(n))
            scores = _get_BLEU_scores(lines_decoded, ref_e_pre, ref_g_pre, n)
            for s in scores:
                output_file.write("\t{0: 1.4f}".format(s))
                #print("\t{}".format(s))
            all_evals.append(scores)

        output_file.write("\n\n")

    output_file.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")

    output_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    args = parser.parse_args()

    main(args)
