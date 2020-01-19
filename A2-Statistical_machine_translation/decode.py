import random
import numpy as np
from math import log
from preprocess import *
from lm_train import *
from log_prob import *
from align_ibm1 import * 

def decode(french, LM, AM):
    
    N = 5           # the maximum number of translations for each word in the sentence
    MAXTRANS = 128; # the maximum number of greedy transformations we perform 
    NUMSWAPS = 2;   # the number of random re-orderings of the words
    
    proposed_french_words = french.split()
    proposed_english_words = []
    
    english_words = AM.keys()
    
    for f_word in proposed_french_words:
        alternatives = []
        for e_word in english_words:
            if f_word in AM[e_word]:
                alternatives = deal_with_alternatives(alternatives, e_word, AM[e_word][f_word], 5)
        if alternatives == []:
            alternatives = [("UNK",0.01)]
        # if alternatives[0][1] == "{1}":
        #     print("alter: {}".format(alternatives))
        #     print("AM: {}".format(AM[e_word][f_word]))
        proposed_english_words.append(alternatives)

    #print(proposed_english_words)
    # Randomly Iterate to find better sentences
    prediction = [word[0] for word in proposed_english_words]
    prediction_score = calc_score(prediction, LM)
    #print("First Prediction:", " ".join([x[0] for x in prediction]), "\t Score:", prediction_score)
    
    for i in range(MAXTRANS):
        #pick new words
        new_guess = []
        for index in range(len(proposed_french_words)):
            new_guess.append(random.choice(proposed_english_words[index]))
        #re_order words
        #Take out start and end
        SS = [new_guess[0]]
        SE = [new_guess[-1]]
        new_guess = new_guess[1:-1]
        random.shuffle(new_guess)
        new_guess = SS + new_guess + SE

        new_guess_prob = calc_score(new_guess, LM)
        
        if new_guess_prob > prediction_score:
            prediction = new_guess
            prediction_score = new_guess_prob
            
    #print()
    #print("Last Prediction:", " ".join([x[0] for x in prediction]), "\t Score:", prediction_score)
    return " ".join([x[0] for x in prediction])
    
    
def deal_with_alternatives(lst, word, prob, num_words = 5):
    if len(lst) < num_words:
        lst.append((word, prob))
        lst = sorted(lst, key=lambda tup: tup[1], reverse=True)
    else:
        if prob > lst[-1][1]:
            lst.append((word, prob))
            lst = sorted(lst, key=lambda tup: tup[1], reverse=True)[:5]
    return lst

def calc_score(e_sentence, LM):
    score = 0
    for i in range(1, len(e_sentence)):
        word_i = e_sentence[i][0]
        word_i_1 = e_sentence[i-1][0]
        if word_i in LM['uni'] and word_i in LM['bi'][word_i_1]:
            # if e_sentence[i][1] == "{1}":
            #     print("first: {}".format(e_sentence[i][1]))
            #     print("second: {}".format(LM['bi'][word_i_1][word_i]))
            #     print("third: ".format(LM['uni'][word_i]))
            score += log(e_sentence[i][1], 2) + log(LM['bi'][word_i_1][word_i], 2) + log(LM['uni'][word_i], 2)
        else:
            return float("-inf")
    return score
