from preprocess import *
import pickle
import os

def lm_train(data_dir, language, fn_LM):
	"""
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM

	INPUTS:

	data_dir	: (string) The top-level directory continaing the data from which
				to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained

	OUTPUT

	LM			: (dictionary) a specialized language model

	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts

	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
	"""

	# TODO: Implement Function
	language_model = {}
	language_model['uni'] = {}
	language_model['bi'] = {}

	# get all file paths inside the datadir
	files_list = os.listdir(data_dir)

	# loop over the files
	for file_path in files_list:
		# check whether the file path is valid
		filename, file_extension = os.path.splitext(file_path)
		if file_extension == '':
			continue
		if file_extension[1:] != language:
			continue

		# build the language model information as long as the file path is valid
		full_path = data_dir + file_path
		file = open(full_path, "r")
		data = file.read().split('\n')
		if '' in data:
			data = data[0:-1]
		file.close()

		for line in data:
			# get tokens list
			tokens = preprocess(line, language).split()

			# store information inside the proper dictionaries
			# NOTE: the last token won't be reached by the idx, thus need to add
			# later in the uni dict
			for i in range(0, len(tokens) - 1):
				token = tokens[i]
				next_token = tokens[i+1]
				# deal with the uni dict
				if token not in language_model['uni'].keys():
					language_model['uni'][token] = 1
				else:
					language_model['uni'][token] += 1

				# deal with the bi dict
				if token not in language_model['bi'].keys():
					language_model['bi'][token] = {}
					language_model['bi'][token][next_token] = 1
				else:
					if next_token not in language_model['bi'][token].keys():
						language_model['bi'][token][next_token] = 1
					else:
						language_model['bi'][token][next_token] += 1

			# add the last token in the uni dict
			last_token = tokens[-1]
			if last_token not in language_model['uni'].keys():
				language_model['uni'][last_token] = 1
			else:
				language_model['uni'][last_token] += 1

	# Save Model
	with open(fn_LM+'.pickle', 'wb') as handle:
	    pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
	    
	return language_model

# if __name__ == "__main__":
# 	data_dir = "/u/cs401/A2_SMT/data/Hansard/Training/"
# 	fn_LM_E = "LM_E"
# 	fn_LM_F = "LM_F"
# 	LM_E = lm_train(data_dir, 'e', fn_LM_E)
# 	LM_F = lm_train(data_dir, 'f', fn_LM_F)
# 	print(LM_E)
# 	print(LM_F)
