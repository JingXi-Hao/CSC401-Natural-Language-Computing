from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
	"""
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model

	OUTPUT:
	AM :			(dictionary) alignment model structure

	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.

			LM['house']['maison'] = 0.5
	"""
	AM = {}

	# Read training data
	eng, fre = read_hansard(train_dir, num_sentences)

	# Initialize AM uniformly
	AM = initialize(eng, fre)

	# Iterate between E and M steps
	for i in range(0, max_iter):
		AM = em_step(AM, eng, fre)

	# Save Model
	with open(fn_AM+'.pickle', 'wb') as handle:
	    pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return AM

	# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
	"""
	Read up to num_sentences from train_dir.

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider


	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.

	Make sure to read the files in an aligned manner.

	OUTPUT:
	(training_eng, training_fren): (tuple of list of list of string) each entry of the tuple
								   is a list of list of string (list of token of preprocessed 
								   sentence) with length equals to num_sentences in English and 
								   French respectively
	"""
	# TODO
	# define output and count
	training_eng = []
	training_fren = []
	count = 0

	# get all filenames under the train_dir
	files_list = os.listdir(train_dir)

	for file_path in files_list:
		# check whether the file path is valid
		filename, file_extension = os.path.splitext(file_path)

		if file_extension == '.e':
			# open and read english and french files
			eng_path = train_dir + file_path
			fren_path = train_dir + filename + '.f'

			eng_file = open(eng_path, "r")
			eng_lines = eng_file.read().split('\n')
			if '' in eng_lines:
				eng_lines = eng_lines[0:-1]
			eng_file.close()

			fren_file = open(fren_path, "r")
			fren_lines = fren_file.read().split('\n')
			if '' in fren_lines:
				fren_lines = fren_lines[0:-1]
			fren_file.close()

			num_loops = 0
			if len(eng_lines) < len(fren_lines):
				num_loops = len(eng_lines)
			else:
				num_loops = len(fren_lines)

			# loop over lines
			for i in range(0, num_loops):
				if count < num_sentences:
					proc_eng_line = preprocess(eng_lines[i], 'e').split()
					proc_fren_line = preprocess(fren_lines[i], 'f').split()

					training_eng.append(proc_eng_line)
					training_fren.append(proc_fren_line)

					count += 1
				else:
					return training_eng, training_fren

	return training_eng, training_fren

def initialize(eng, fre):
	"""
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.

	INPUTS:
	eng: (list of list of string) num_sentences of tokenized preprocessed sentences in English
	fre: (list of list of string) num_sentences of tokenized preprocessed sentences in French

	OUTPUT:
	alignment_model: (dictionary of dictionary) each possible alignment with its probability 
	"""
	# TODO

	# define output and special cases
	alignment_model = {'SENTSTART': {'SENTSTART': 1}, 'SENTEND': {'SENTEND': 1}}

	align_info = {}
	for i in range(0, len(eng)):
		eng_sen = eng[i][1:-1]
		fre_sen = fre[i][1:-1]

		for eng_token in eng_sen:
			if eng_token not in align_info:
				align_info[eng_token] = []
			for fre_token in fre_sen:
				if fre_token not in align_info[eng_token]:
					align_info[eng_token].append(fre_token)
			# set is expensive here, cause unefficiency
			#align_info[eng_token].extend(fre_sen)
			#align_info[eng_token] = list(set(align_info[eng_token]))

	for token in align_info.keys():
		fren_list = align_info[token]
		alignment_model[token] = {}
		for fren_token in fren_list:
			if len(fren_list) != 0:
				alignment_model[token][fren_token] = 1 / float(len(fren_list))

	return alignment_model

def em_step(t, eng, fre):
	"""
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	INPUTS:
	t: (dictionary of dictionary) previous AM model
	eng: (list of list of string) num_sentences of tokenized preprocessed sentences in English
	fre: (list of list of string) num_sentences of tokenized preprocessed sentences in French

	OUTPUT:
	AM: (dictionary of dictionary) updated AM model after one em step 
	"""
	# TODO
	# define output and special case
	AM = {'SENTSTART': {'SENTSTART': 1}, 'SENTEND': {'SENTEND': 1}}

	# initialize tcount and total
	tcount = {}
	total = {}

	for eng_token in t:
		if (eng_token == 'SENTSTART') or (eng_token == 'SENTEND'):
			continue
		# key is unique so no need to check
		total[eng_token] = 0
		tcount[eng_token] = {}
		for fren_token in t[eng_token]:
			tcount[eng_token][fren_token] = 0

	# compute tcount and count for each given e, f
	for i in range(0, len(eng)):
		# make sure each token is unique
		eng_token_list = eng[i][1:-1]
		fren_token_list = fre[i][1:-1]

		unique_eng_tokens = list(set(eng_token_list))
		unique_fren_tokens = list(set(fren_token_list))

		eng_token_counts = {}
		fre_token_counts = {}
		for token in eng_token_list:
			if token not in eng_token_counts:
				eng_token_counts[token] = eng_token_list.count(token)

		for token in fren_token_list:
			if token not in fre_token_counts:
				fre_token_counts[token] = fren_token_list.count(token)

		for fren_token in unique_fren_tokens:
			denom = 0
			for eng_token in unique_eng_tokens:
				denom += t[eng_token][fren_token] * fre_token_counts[fren_token]
			if denom == 0:
				continue
			for eng_token in unique_eng_tokens:
				tcount[eng_token][fren_token] += (t[eng_token][fren_token] * 
					fre_token_counts[fren_token] * 
					eng_token_counts[eng_token]) / float(denom)
				total[eng_token] += (t[eng_token][fren_token] * 
					fre_token_counts[fren_token] * 
					eng_token_counts[eng_token]) / float(denom)

	for eng_token in total:
		AM[eng_token] = {}
		for fren_token in tcount[eng_token]:
			if (total[eng_token] !=  0) and (tcount[eng_token][fren_token] != 0):
				AM[eng_token][fren_token] = tcount[eng_token][fren_token] / float(total[eng_token])
			# else:
			# 	AM[eng_token][fren_token] = 0 # prob zero --> no alignment

	return AM

# if __name__ == "__main__":
# 	# align_ibm1(train_dir, num_sentences, max_iter, fn_AM)
# 	train_dir = "/u/cs401/A2_SMT/data/Hansard/Training/"
# 	num_sentences = 1000
# 	max_iter = 10 # 2, 5, 10, 20, 50
# 	fn_AM = 'am_test_1000'

# 	result = align_ibm1(train_dir, num_sentences, max_iter, fn_AM)
# 	print(result)
