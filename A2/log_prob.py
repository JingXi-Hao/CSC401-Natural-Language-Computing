from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
	"""
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing

	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary

	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""

	#TODO: Implement by student.
	# obtain tokens (words) list
	tokens = sentence.split()
	log_prob = 0

	# loop over tokens list, and compute the log probability for each token, then
	# sum up to get log probability of a sentence
	for i in range(1, len(tokens)):
		cur_token = tokens[i]
		prev_token = tokens[i-1]

		# define numerator
		if (prev_token in LM['bi'].keys()) and (cur_token in LM['bi'][prev_token].keys()):
			num_count = LM['bi'][prev_token][cur_token]
		else:
			num_count = 0

		# define denominator
		if prev_token in LM['uni'].keys():
			den_count = LM['uni'][prev_token]
		else:
			den_count = 0

		# compute numerator and denominator
		if smoothing == False:
			num = num_count
			den = den_count
		else:
			num = num_count + delta
			den = den_count + delta * vocabSize
		
		if (num != 0) and (den != 0):
			log_prob = log_prob + log(num / float(den), 2)
		else:
			return float("-inf")
	        
	return log_prob

# if __name__ == "__main__":
# 	sentence = "SENTSTART this is a test case ! SENTEND"
# 	fn_LM = 'LM_E'

# 	with open(fn_LM+'.pickle', 'rb') as handle:
# 		LM = pickle.load(handle)

# 	smoothing = False
# 	delta = 0
# 	vocabSize = len(LM["uni"])
# 	log_prob = log_prob(sentence, LM, smoothing, delta, vocabSize)
# 	print(log_prob)
