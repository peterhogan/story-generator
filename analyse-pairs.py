import nltk
import numpy as np
import argparse
from time import time
from collections import Counter

np.set_printoptions(threshold=np.inf)

# Function to read nice byte sizes:
def size_format(x, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(x) < 1024.0:
            return "%3.1f%s%s" % (x, unit, suffix)
        x /= 1024.0
    return "%.1f%s%s" % (x, 'Yi', suffix)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Read pairs of words in a file and construct a markov chain of the words.")
parser.add_argument("textfile", help="text file to read")
args = parser.parse_args()

inputfile = args.textfile

# start timer
start = time()

with open(inputfile) as textfile:
    lines = textfile.read().splitlines()

alltext = " ".join(lines)
justwords = nltk.regexp_tokenize(alltext, r"\w+")
alltokens = nltk.word_tokenize(alltext)
postags = nltk.pos_tag(alltokens)

uniquewords = list(set(justwords))

indexedwords = []
for i in range(len(uniquewords)):
    indexedwords.append((uniquewords[i],i))

wdict = dict(indexedwords)

arraylist = []

for i in range(len(justwords)):
    try:
        fst = justwords[i]
        snd = justwords[i+1]
        arraylist.append((wdict[fst],wdict[snd]))

    except IndexError:
        pass

wordsdims = (len(uniquewords),len(uniquewords))
wordarray = np.zeros(wordsdims)
counts = Counter(arraylist)

for x in range(len(uniquewords)):
    for y in range(len(uniquewords)):
        wordarray[x][y] = counts[(x,y)]/len(justwords)

for i in range(len(uniquewords)):
    reference = wordarray[(wdict['the'],i)]
    if reference > 0:
        matchword = [word for word, index in indexedwords if index == i][0]
        print(matchword+":",reference)
    else:
        pass


#print(np.sum(wordarray))
print("Time taken:",time() - start)
