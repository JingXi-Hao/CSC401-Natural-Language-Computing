import numpy as np
import sys
import argparse
import os
import json

# add new imports
import re
import string
import csv
#from tqdm import tqdm

# constant variables
# this list of words come from a1 handout, define slang regex pattern
slangList_1000654188 = ["smh", "fwb", "lmfao", "lmao", "lms", "tbh", "ro", "wtf", "bff", "wyd", "lylc", 
    "brb", "atm", "imao", "sml", "btw", "bw", "imho", "fyi", "ppl", "sob", "ttyl", "imo", "ltr", 
    "thx", "kk", "omg", "omfg", "ttys", "afn", "bbs", "cya", "ez", "f2f", "gtr", "ic", "jk", "k", 
    "ly", "ya", "nm", "np", "plz", "ru", "so", "tc", "tmi", "ym", "ur", "u", "sol", "fml"]
slangPattern_1000654188 = r'|'.join(slangList_1000654188)

# csv files
bngDir_1000654188 = "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"
rwDir_1000654188 = "/u/cs401/Wordlists/Ratings_Warriner_et_al.csv"

# read bng csv file and save the required information in a dictionary, where key is a word
# and item is a list -- [AoA', 'IMG', 'FAM'], save corresponding information at the fixed
# index
bngCSV_1000654188 = open(bngDir_1000654188, "r")
bngFile_1000654188 = csv.reader(bngCSV_1000654188)
bngInfo_1000654188 = {}
skipFirstLine_1000654188 = False
for line_1000654188 in bngFile_1000654188:
    if skipFirstLine_1000654188 == True:
        wordKey_1000654188 = line_1000654188[1]
        if wordKey_1000654188 != "":
            aoa_1000654188 = float(line_1000654188[3])
            img_1000654188 = float(line_1000654188[4])
            fam_1000654188 = float(line_1000654188[5])
            if wordKey_1000654188 not in bngInfo_1000654188.keys():
                bngInfo_1000654188[wordKey_1000654188] = []
                bngInfo_1000654188[wordKey_1000654188].append(aoa_1000654188)
                bngInfo_1000654188[wordKey_1000654188].append(img_1000654188)
                bngInfo_1000654188[wordKey_1000654188].append(fam_1000654188)
            else:
                bngInfo_1000654188[wordKey_1000654188][0] += aoa_1000654188
                bngInfo_1000654188[wordKey_1000654188][1] += img_1000654188
                bngInfo_1000654188[wordKey_1000654188][2] += fam_1000654188
    else:
        skipFirstLine_1000654188 = True
# build dictionary based on rw csv file, where key is a word and item is a list -- ['V', 'A', 'D']
rwCSV_1000654188 = open(rwDir_1000654188, "r")
rwFile_1000654188 = csv.reader(rwCSV_1000654188)
rwInfo_1000654188 = {}
skipFirstLine_1000654188 = False
for line_1000654188 in rwFile_1000654188:
    if skipFirstLine_1000654188 == True:
        wordKey_1000654188 = line_1000654188[1]
        if wordKey_1000654188 != "":
            v_1000654188 = float(line_1000654188[2])
            a_1000654188 = float(line_1000654188[5])
            d_1000654188 = float(line_1000654188[8])
            if wordKey_1000654188 not in rwInfo_1000654188.keys():
                rwInfo_1000654188[wordKey_1000654188] = []
                rwInfo_1000654188[wordKey_1000654188].append(v_1000654188)
                rwInfo_1000654188[wordKey_1000654188].append(a_1000654188)
                rwInfo_1000654188[wordKey_1000654188].append(d_1000654188)
            else:
                rwInfo_1000654188[wordKey_1000654188][0] += v_1000654188
                rwInfo_1000654188[wordKey_1000654188][1] += a_1000654188
                rwInfo_1000654188[wordKey_1000654188][2] += d_1000654188
    else:
        skipFirstLine_1000654188 = True

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # TODO: your code here

    # define feats here
    feats = np.zeros((1, 173))

    # if the string consists of only spaces or empty string, then return feats
    comment = re.sub(r"^\s+$", "", comment)
    if comment == "":
        return feats

    # modify comment to add space at the front and end
    comment = " {} ".format(comment)

    # below here, we extract all features for given comment as required
    # number of first-person pronouns -- first person based on a1 handout
    # I, me, my, mine, we, us, our, ours
    fppList = re.compile(r"(?<=\s)\b(i|me|my|mine|we|us|our|ours)\b(?=\/)").findall(comment)
    feats[0][0] = len(fppList)

    # number of second-person pronouns
    # you, your, yours, u, ur, urs
    sppList = re.compile(r"(?<=\s)+\b(you|your|yours|u|ur|urs)\b(?=\/)").findall(comment)
    feats[0][1] = len(sppList)

    # number of third-person pronouns
    # he, him, his, she, her, hers, it, its, they, them, their, theirs
    tppList = re.compile(r"(?<=\s)\b(he|him|his|she|her|hers|it|its|they|them|their|theirs)\b(?=\/)").findall(comment)
    feats[0][2] = len(tppList)

    # number of coordinating conjunctions
    ccList = re.compile(r"(?<=\/)\b(CC)\b(?=\s+)").findall(comment)
    feats[0][3] = len(ccList)

    # number of past-tense verbs
    ptvList = re.compile(r"(?<=\/)\b(VBD)\b(?=\s+)").findall(comment)
    feats[0][4] = len(ptvList)

    # number of future-tense verbs
    # 'll, will, gonna, going+to+VB
    ftvList = re.compile(r"\b(\'ll|will|gonna)\b").findall(comment)
    ftvList2 = re.compile(r"(?<=\s)\b(going\/VBG)(\s+to\/TO\s+)([\w]+\/VB)\b(?=\s+)").findall(comment)
    feats[0][5] = len(ftvList) + len(ftvList2)

    # number of commas
    cList = re.compile(r"(?<=\s)(,\/,)(?=\s+)").findall(comment)
    feats[0][6] = len(cList)

    # number of multi-character punctuation tokens
    mcptList = re.compile(r"(?<=\s)[\\!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]{2,}(?=\/)").findall(comment)
    feats[0][7] = len(mcptList)

    # number of common nouns
    cnList = re.compile(r"(?<=\/)\b(NN|NNS)\b(?=\s+)").findall(comment)
    feats[0][8] = len(cnList)

    # number of proper nouns - NNP, NNPS
    pnList = re.compile(r"(?<=\/)\b(NNP|NNPS)\b(?=\s+)").findall(comment)
    feats[0][9] = len(pnList)

    # number of adverbs - RB, RBR, RBS
    aList = re.compile(r"(?<=\/)\b(RB|RBR|RBS)\b(?=\s+)").findall(comment)
    feats[0][10] = len(aList)

    # number of wh- words -- WDT, WP, WP$, WRB
    whList = re.compile(r"(?<=\/)\b(WDT|WP|WP\$|WRB)\b(?=\s+)").findall(comment)
    feats[0][11] = len(whList)

    # number of slang acronyms
    sList = re.compile(r"(?<=\s)\b(" + slangPattern_1000654188 + r")\b(?=\/)").findall(comment)
    feats[0][12] = len(sList)

    # number of words in uppercase (>= 3 letters long)
    uList = re.compile(r"(?<=\s)[A-Z]{3,}(?=\/)").findall(comment)
    feats[0][13] = len(uList)

    # average length of sentences, in tokens
    sentences = comment.split('\n')
    totalSen = len(sentences)
    # if empty string exists, minus one
    if "" in sentences:
        totalSen = len(sentences) - 1
    allTokensList = re.compile(r"(?<=\s)(\S+)(?=\s+)").findall(comment)
    totalTok = len(allTokensList)
    #print("total sen: " + str(totalSen))
    #print("totalTok: " + str(totalTok))
    if totalSen != 0:
        feats[0][14] = totalTok / float(totalSen)

    # average length of tokens, excluding punctuation-only tokens, in characters
    partialTokenList = re.compile(r"(?<=\s)([\S]*[\w]+[\S]*\/[\S]*)(?=\s+)").findall(comment)
    #print("comment: " + str(comment))
    #print("ptl: " + str(partialTokenList))
    totalCharTok = len(partialTokenList)
    totalChar = 0
    for t in partialTokenList:
        totalChar += len(t)
    if totalCharTok != 0:
        feats[0][15] = totalChar / float(totalCharTok)

    # number of sentences
    feats[0][16] = totalSen

    # process bng and rw information and store needed information into feats matrix
    wordTokenList = re.compile(r"(?<=\s)([\w]+)(?=\/[A-Z]+[$]*\s+)").findall(comment)
    aoaList = []
    imgList = []
    famList = []
    vList = []
    aList = []
    dList = []

    for wordToken in wordTokenList:
        if wordToken in bngInfo_1000654188.keys():
            aoaList.append(bngInfo_1000654188[wordToken][0])
            imgList.append(bngInfo_1000654188[wordToken][1])
            famList.append(bngInfo_1000654188[wordToken][2])
        if wordToken in rwInfo_1000654188.keys():
            vList.append(rwInfo_1000654188[wordToken][0])
            aList.append(rwInfo_1000654188[wordToken][1])
            dList.append(rwInfo_1000654188[wordToken][2])
    
    # average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    # average of IMG from Bristol, Gilhooly, and Logie norms
    # average of FAM from Bristol, Gilhooly, and Logie norms
    # Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    # Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    # Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
    if len(aoaList) != 0:
        aoaList = np.array(aoaList)
        feats[0][17] = np.mean(aoaList)
        feats[0][20] = np.std(aoaList)
    if len(imgList) != 0:
        imgList = np.array(imgList)
        feats[0][18] = np.mean(imgList)
        feats[0][21] = np.std(imgList)
    if len(famList) != 0:
        famList = np.array(famList)
        feats[0][19] = np.mean(famList)
        feats[0][22] = np.std(famList)
    
    # average of V.Mean.Sum from Warringer norms
    # average of A.Mean.Sum from Warringer norms
    # average of D.Mean.Sum from Warringer norms
    # standard deviation of V.Mean.Sum from Warringer norms
    # standard deviation of A.Mean.Sum from Warringer norms
    # standard deviation of D.Mean.Sum from Warringer norms
    if len(aList) != 0:
        aList = np.array(aList)
        feats[0][23] = np.mean(aList)
        feats[0][26] = np.std(aList)
    if len(vList) != 0:
        vList = np.array(vList)
        feats[0][24] = np.mean(vList)
        feats[0][27] = np.std(vList)
    if len(dList) != 0:
        dList = np.array(dList)
        feats[0][25] = np.mean(dList)
        feats[0][28] = np.std(dList)

    # strip the extra space added at the beginning and the end
    comment = re.sub(r"^\s", "", comment)
    comment = re.sub(r"\s$", "", comment)

    return feats

def main( args ):

    data = json.load(open(args.input))
    #print(type(data)) -- data here is a list of dicts
    feats = np.zeros( (len(data), 173+1) )

    # TODO: your code here
    # read all needed files
    liwcDir = "/u/cs401/A1/feats/"
    partialIDPath = "_IDs.txt"
    partialFeatsPath = "_feats.dat.npy"

    rightIDFile = open(liwcDir + "Right" + partialIDPath, "r")
    rightIDs = rightIDFile.read().split("\n")
    rightFeats = np.load(liwcDir + "Right" + partialFeatsPath)

    altIDFile = open(liwcDir + "Alt" + partialIDPath, "r")
    altIDs = altIDFile.read().split("\n")
    altFeats = np.load(liwcDir + "Alt" + partialFeatsPath)

    centerIDFile = open(liwcDir + "Center" + partialIDPath, "r")
    centerIDs = centerIDFile.read().split("\n")
    centerFeats = np.load(liwcDir + "Center" + partialFeatsPath)

    leftIDFile = open(liwcDir + "Left" + partialIDPath, "r")
    leftIDs = leftIDFile.read().split("\n")
    leftFeats = np.load(liwcDir + "Left" + partialFeatsPath)

    for i in range(0, len(data)):
        comment = data[i]['body']
        commentFile = data[i]['cat']
        commentID = data[i]['id']
        
        # store first 29 features extracted first
        commentFeats = extract1(comment)
        feats[i][0:29] = commentFeats[0][0:29]

        # store 30 - 173 features and set the 174th feature with 
        # 0: Left, 1: Center, 2: Right, 3: Alt
        if commentFile == "Right":
            rightIndex = rightIDs.index(commentID)
            feats[i][29:173] = rightFeats[rightIndex, :]
            feats[i][173] = 2
        
        if commentFile == "Alt":
            altIndex = altIDs.index(commentID)
            feats[i][29:173] = altFeats[altIndex, :]
            feats[i][173] = 3
        
        if commentFile == "Center":
            centerIndex = centerIDs.index(commentID)
            feats[i][29:173] = centerFeats[centerIndex, :]
            feats[i][173] = 1
        
        if commentFile == "Left":
            leftIndex = leftIDs.index(commentID)
            feats[i][29:173] = leftFeats[leftIndex, :]
            feats[i][173] = 0

    #print(feats)
    #print(np.argwhere(np.isnan(feats)))

    np.savez_compressed( args.output, feats )

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

