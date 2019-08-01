import re

# add new imports
import string

def preprocess(in_sentence, language):
	""" 
	This function preprocesses the input text according to language-specific rules.
	Specifically, we separate contractions according to the source language, convert
	all tokens to lower-case, and separate end-of-sentence punctuation 

	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
	"""
	# TODO: Implement Function
	start_tag = 'SENTSTART'
	end_tag = 'SENTEND'
	out_sentence = ''
	in_sentence_copy = in_sentence

	# step 1: remove newline character, convert into lower case letters and add sentence tags for it
	in_sentence = in_sentence.replace("\n", " ").replace("\r", " ")
	in_sentence = '{} {} {}'.format(start_tag, in_sentence.lower(), end_tag)
	in_sentence = re.sub(r"\s+", " ", in_sentence) 
	in_sentence = in_sentence.strip()

	# step 2: separate sentence-final punctuation
	pattern = re.compile(r"([\.?!])")
	in_sentence = re.sub(pattern, r" \1 ", in_sentence)
	# in_sentence = re.sub(r"\s+", " ", in_sentence)
	# in_sentence = in_sentence.strip()

	# step 3: separate commas, colons, semicolons, parentheses, mathematical operators,
	# and quotation marks
	pattern = re.compile(r"([,:;\(\)\+\-<>=\"\`])")
	in_sentence = re.sub(pattern, r" \1 ", in_sentence)
	# in_sentence = re.sub(r"\s+", " ", in_sentence)
	# in_sentence = in_sentence.strip()

	# step 5 is included in step 3
	# step 5: separate dashes between parentheses
	# pattern = re.compile(r"([\(])([-])([\)])")
	# in_sentence = re.sub(pattern, r" \1 \2 \3 ", in_sentence)
	# in_sentence = re.sub(r"\s+", " ", in_sentence)
	# in_sentence = in_sentence.strip()

	# step 7: separate clitics, need to consider language
	in_sentence = " {} ".format(in_sentence)
	if language == 'f':
		# Singular definite article
		pattern = re.compile(r"(?<=\s)(l\')(?=[\w]+)")
		in_sentence = re.sub(pattern, r" \1 ", in_sentence)

		# que
		pattern = re.compile(r"(?<=\s)(qu\')(?=[\w]+)")
		in_sentence = re.sub(pattern, r" \1 ", in_sentence)

		# Conjunctions: puisque and lorsque
		pattern = re.compile(r"(?<=\')(on|il)(?=\s)")
		in_sentence = re.sub(pattern, r" \1 ", in_sentence)

		# Single-consonant words ending in e-`muet'
		pattern = re.compile(r"(?<=\s)(b|c|d|f|g|h|j|k|m|n|p|q|r|s|t|v|x|z)(\')([\w]+)(?=\s)")
		specialCases = ['abord', 'accord', 'ailleurs', 'habitude']
		def processCon(match):
			if match.group(1) == 'd':
				if match.group(3) in specialCases:
					return match.group(1) + match.group(2) + match.group(3)
				else:
					return match.group(1) + match.group(2) + ' ' + match.group(3)
			else:
				return match.group(1) + match.group(2) + ' ' + match.group(3)
		in_sentence = re.sub(pattern, processCon, in_sentence)

	else:
		# deal with t', y'
	    pattern = re.compile(r"(?<=\s)(t|y)(\')(?=[\w]+)")
	    in_sentence = re.sub(pattern, r" \1\2 ", in_sentence)
	    # deal with 'd, 'n, 've, 're, 'll, 'm, 're, 's, 't
	    pattern1 = re.compile(r"([\w]+)(\'d|\'n|\'ve|\'re|\'ll|\'m|\'s|\'t)(?=\s)")
	    cliticList = ["'d", "'n", "'ve", "'re", "'ll", "'m", "'s"]
	    def splitMatch(match):
	        if match.group(2).lower() in cliticList:
	            return match.group(1) + " " + match.group(2) + " "
	        if match.group(2).lower() == "'t":
	            return match.group(1)[0:-1] + " " + match.group(1)[-1] + match.group(2) + " "
	    in_sentence = re.sub(pattern1, splitMatch, in_sentence)
	    # deal with s'
	    pattern2 = re.compile(r"(?<=s)(\')(?=\s)")
	    in_sentence = re.sub(pattern2, r" \1 ", in_sentence)

	in_sentence = re.sub(r"\s+", " ", in_sentence)
	in_sentence = in_sentence.strip()

	# assign out_sentence
	out_sentence = in_sentence

	return out_sentence

# if __name__ == "__main__":
# 	print(preprocess('t.<this is a -- (-)test!', 'e'))
# 	print(preprocess("je t'aime", 'f'))
