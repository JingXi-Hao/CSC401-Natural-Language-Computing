from log_prob import *
from preprocess import *
import os

def preplexity(LM, test_dir, language, smoothing = False, delta = 0):
    """
    Computes the preplexity of language model given a test corpus

    INPUT:

    LM :        (dictionary) the language model trained by lm_train
    test_dir :  (string) The top-level directory name containing data
                e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    language : `(string) either 'e' (English) or 'f' (French)
    smoothing : (boolean) True for add-delta smoothing, False for no smoothing
    delta :     (float) smoothing parameter where 0<delta<=1
    """

    files = os.listdir(test_dir)
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])

    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue
        
        opened_file = open(test_dir+ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language)
            tpp = log_prob(processed_line, LM, smoothing, delta, vocab_size)
            # if tpp == float("-inf"):
            #     print("-inf occur")
            
            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())
        opened_file.close()
    if N > 0:
        pp = 2**(-pp/N)
    return pp

#test
test_LM_E = lm_train("/u/cs401/A2_SMT/data/Hansard/Training/", "e", "LM_E")
test_LM_F = lm_train("/u/cs401/A2_SMT/data/Hansard/Training/", "f", "LM_F")

values = [0.0001, 0.00014, 0.0005, 0.0008, 0.001, 0.005, 0.008, 0.01, 0.05, 0.08, 0.1, 0.5, 0.7, 0.9, 1.0]
print("### Perplexity For English Model ### \n")
print("delta = {} and perplexity = {}".format(0.0, 
        preplexity(test_LM_E, "/u/cs401/A2_SMT/data/Hansard/Testing/", "e")))
for delta in values:
    print("delta = {} and perplexity = {}".format(delta, 
        preplexity(test_LM_E, "/u/cs401/A2_SMT/data/Hansard/Testing/", "e", True, delta)))

print("### Perplexity For French Model ### \n")
print("delta = {} and perplexity = {}".format(0.0, 
        preplexity(test_LM_F, "/u/cs401/A2_SMT/data/Hansard/Testing/", "f")))
for delta in values:
    print("delta = {} and perplexity = {}".format(delta, 
        preplexity(test_LM_F, "/u/cs401/A2_SMT/data/Hansard/Testing/", "f", True, delta)))
