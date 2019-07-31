import math

def BLEU_score(candidate, references, n, brevity=False):
	"""
	Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
	n=1 unigram
	n=2 bigram
	... and so on

	DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.

	INPUTS:
	sentence :	(string) Candidate sentence. "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.


	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""

	#TODO: Implement by student.
	# this function does not compute the final BLEU score, it just compute one
	# level of it
	can_tokens = candidate.split()#[1:-1] # do not ignore SENTSTART and SENTEND

	# compute brevity when is assigned to be True
	if brevity == True and n == 1:
		diffs = []
		for ref in references:
			ref_tokens = ref.split()
			diffs.append(abs(len(ref_tokens) - len(can_tokens)))

		idx = diffs.index(min(diffs))
		best_match = len(references[idx].split())
		brevity = best_match / float(len(can_tokens))
		if brevity < 1:
			BP = 1
		else:
			BP = math.exp(1 - brevity)

	# compute precisions for n-level
	count = 0
	num_tokens = 0
	for i in range(0, len(can_tokens)):
		idx = i + n - 1
		if idx < len(can_tokens):
			token = ' '.join(can_tokens[i : idx+1])
			num_tokens += 1
			# update the count
			if n == 1:
				if token in (' '.join(references)).split():
					count += 1
			else: 
				for ref in references:
					if token in ref:
						count += 1
						break
		else:
			break
	#precision = count / float(len(can_tokens))
	precision = count / float(num_tokens)

	# compute bleu_score
	if brevity == True and n == 1:
		bleu_score = precision * BP
	else:
		bleu_score = precision
	        
	return bleu_score
