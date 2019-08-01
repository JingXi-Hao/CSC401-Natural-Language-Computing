import sys
import argparse
import os
import json

# add imports
import string
import html
import re
import spacy
#from tqdm import tqdm

indir = '/u/cs401/A1/data/';

# add new directories
abbrevDir_1000654188 = '/u/cs401/Wordlists/abbrev.english';
pnAbbrevDir_1000654188 = '/u/cs401/Wordlists/pn_abbrev.english';
stopwordsDir_1000654188 = '/u/cs401/Wordlists/StopWords';

# read in all the abbreviations
abbrevFile_1000654188 = open(abbrevDir_1000654188, "r")
abbrevList_1000654188 = abbrevFile_1000654188.read().split('\n')
pnAbbrevFile_1000654188 = open(pnAbbrevDir_1000654188, "r")
pnAbbrevList_1000654188 = pnAbbrevFile_1000654188.read().split('\n')
abbrevWords_1000654188 = pnAbbrevList_1000654188 + list(set(abbrevList_1000654188) - set(pnAbbrevList_1000654188))
abbrevWords_1000654188 = [word_1000654188.upper() for word_1000654188 in abbrevWords_1000654188]

abbPattern_1000654188 = ""
for abb_1000654188 in abbrevWords_1000654188:
    if abb_1000654188 != "":
        if abb_1000654188 == "E.G.":
            abbPattern_1000654188 = abbPattern_1000654188 + r"(?<=\sE\.G)|"
        elif abb_1000654188 == "I.E.":
            abbPattern_1000654188 = abbPattern_1000654188 + r"(?<=\sI\.E)|"
        else:
            abbPattern_1000654188 = abbPattern_1000654188 + r"(?<=\s{})|".format(abb_1000654188[0:-1])
abbPattern_1000654188 = r"({})".format(abbPattern_1000654188[0:-1])
#print(abbPattern)

# read in all stop words
swFile_1000654188 = open(stopwordsDir_1000654188, "r")
swList_1000654188 = swFile_1000654188.read().split("\n")

swPattern_1000654188 = ""
for sw_1000654188 in swList_1000654188:
    if sw_1000654188 != "":
        swPattern_1000654188 = swPattern_1000654188 + r"{}|".format(sw_1000654188)
swPattern_1000654188 = r"({})".format(swPattern_1000654188[0:-1])

# define nlp
nlp_1000654188 = spacy.load('en', disable = ['parser','ner'])

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = ''
    commentCopy = comment

    if 1 in steps:
        # remove all newline characters
        comment = comment.replace("\n", " ").replace("\r", " ")
        comment = re.sub(r"\s+", " ", comment)
        comment = comment.strip()
    if 2 in steps:
        # replace html character codes with their ASCII equivalent
        comment = html.unescape(comment)
    if 3 in steps:
        # remove URLs start with http or www with case insensitivity
        urlPattern = re.compile(r"\]?(https?:\/\/|www\.)[a-z0-9\\!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]+",
            re.IGNORECASE)
        comment = re.sub(urlPattern, "", comment)
    if 4 in steps:
        pattern = re.compile(r"(?<=\s)(\S+)(?=\s+)")
        pattern2 = re.compile(abbPattern_1000654188 + r"([.])", re.IGNORECASE)
        # split any abbreviation with anything following it, however for e.g. and i.e., we also need 
        # to split by anything connected to it at the front
        comment = re.sub(pattern2, r"\2 ", comment)
        comment = re.sub(r"(E\.G\.|e\.g\.|I\.E\.|i\.e\.)", r" \1 ", comment)
        comment = " {} ".format(comment)
        def processMatch(match):
            token = match.group(0)
            if token.upper() in abbrevWords_1000654188:
                token = " {} ".format(token)
            else:
                # no abbreviation matched, keep the clitics and split everything else
                token = re.sub(r"((\W*\'\W+)|[\\!\"#$%&\()*+,-./:;<=>?@[\]^_`{|}~]+)", r" \1 ", token)
            return token
        comment = re.sub(pattern, processMatch, comment)
        comment = re.sub(r"\s+", " ", comment)
        comment = comment.strip()
        #print(comment)
    if 5 in steps:
        # based on the clictis files provided, we have 'd, 'n, 've, 're, 'll, 'm, 're, 's, t', y'
        # deal with t', y'
        pattern3 = re.compile(r"(?<=\s)(t|y)(\')", re.IGNORECASE)
        comment = re.sub(pattern3, r" \1\2 ", comment)
        # deal with 'd, 'n, 've, 're, 'll, 'm, 're, 's, 't
        pattern = re.compile(r"([\w]+)(\'d|\'n|\'ve|\'re|\'ll|\'m|\'s|\'t)", re.IGNORECASE)
        cliticList = ["'d", "'n", "'ve", "'re", "'ll", "'m", "'s"]
        def splitMatch(match):
            if match.group(2).lower() in cliticList:
                return match.group(1) + " " + match.group(2) + " "
            if match.group(2).lower() == "'t":
                return match.group(1)[0:-1] + " " + match.group(1)[-1] + match.group(2) + " "
        comment = re.sub(pattern, splitMatch, comment)
        # deal with s'
        pattern2 = re.compile(r"((?<=s))(\')(?=\s)", re.IGNORECASE)
        comment = re.sub(pattern2, r" \2 ", comment)
        # remove extra white spaces
        comment = re.sub(r"\s+", " ", comment)
        comment = comment.strip()
        #print(comment)
    if 6 in steps:
        # based on tutorial 2: add tag for each word using spacy
        tokens = comment.strip().split()
        doc = spacy.tokens.Doc(nlp_1000654188.vocab, words=tokens)
        doc = nlp_1000654188.tagger(doc)
        # add tag information to comment string
        newComment = ''
        for tIndex in range(0, len(doc)):
            token = doc[tIndex]
            if tIndex != (len(doc) - 1):
                newComment = newComment + str(token.text) + '/' + str(token.tag_)  + ' '
            else:
                newComment = newComment + str(token.text) + '/' + str(token.tag_) 
        comment = newComment
        #print (comment)
    if 7 in steps:
        # remove stopwords and tags if tags are added
        # the special character here is to make sure every tag gonna be removed with stop words
        # '$' added since saw an example as "PRP$"
        pattern = re.compile(r"\b"+ swPattern_1000654188 + r"\b" + r"(\/[A-Z]+[$]*\s+)*", re.IGNORECASE)
        comment = re.sub(pattern, " ", comment)
        # remove extra spaces
        comment = re.sub(r"\s+", " ", comment)
        comment = comment.strip()
        #print(comment)
    if 8 in steps:
        # based on the instructor's response on Piazza, we can assume that steps 6, 8, 9 will 
        # run together, therefore, use previous 'doc' and tags are added in comment
        # process each token replace the token itself with the token.lemma_
        newComment = ""
        for token in doc:
            textInComment = str(token.text) + "/" + str(token.tag_)
            # lemma does not start with a '-'
            if token.lemma_[0] != '-':
                # token has not been removed
                if textInComment in comment:
                    newComment = newComment + str(token.lemma_) + "/" + str(token.tag_) + " "
            else:
                # both lemma and token starts with '-', then replace
                if token.text[0] == '-':
                    # token has not been removed
                    if textInComment in comment:
                        newComment = newComment + str(token.lemma_) + "/" + str(token.tag_) + " "
                else:
                    # token has not been removed
                    if textInComment in comment:
                        newComment = newComment + str(token.text) + "/" + str(token.tag_) + " "

        comment = newComment.strip()   
        #print("comment: " + comment)
    if 9 in steps:
        # based on the instructor's response on Piazza, we can assume that steps 6, 8, 9 will 
        # run together, therefore, no need to consider case: text without tag
        # add a newline between each sentence, since the tag for sentence termination punctuation
        # is '.', therefore, only check for that tag
        pattern = re.compile(r"(?<=\/)([.])(?=\s)")
        # add extra space before and after the comment, if there is end sentence character at the end, 
        # the extra space helps to add newline character
        comment = " {} ".format(comment)
        comment = re.sub(pattern, r"\1\n", comment)
        # remove extra space at the beginning and the end
        comment = re.sub(r"^\s", "", comment)
        comment = re.sub(r"\s$", "", comment)
        #print(comment)
    if 10 in steps:
        # convert text to lowercase
        if 6 not in steps:
            comment = comment.lower()
        else:
            comment = " {} ".format(comment)
            def convertLowerCase(match):
                return match.group(0).lower()
            comment = re.sub(r"(?<=\s)([\S]*[\w]+[\S]*)(?=\/)", convertLowerCase, comment)
            comment = re.sub(r"^\s", "", comment)
            comment = re.sub(r"\s$", "", comment)
        #print(comment)

    modComm = comment    
    return modComm

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            index = args.ID[0] % len(data)
            iterations = 0

            while (iterations < args.max):
                # TODO: read those lines with something like `j = json.loads(line)`
                rawLine = json.loads(data[index])
                # TODO: choose to retain fields from those lines that are relevant to you
                line = {}
                line["id"] = rawLine["id"]
                line["body"] = rawLine["body"]
                # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
                line["cat"] = file 
                # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
                bodyProcessed = preproc1(line["body"])
                # TODO: replace the 'body' field with the processed text
                line["body"] = bodyProcessed
                # TODO: append the result to 'allOutput'
                allOutput.append(line)

                # update index and iterations
                iterations += 1
                index += 1
                if index > (len(data) - 1):
                    index = 0
            
    #print(len(allOutput))

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    # convert args.max into integer
    args.max = int(args.max)

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
        
    main(args)
