'''
Class Activity 3
'''
import os
import glob
import re
import nltk

# Note: first make sure that your current working directory is the code folder!
path = '../summarization_dataset_duc_2004/test_docs'
gspath = '../summarization_dataset_duc_2004/gold_summaries'


# all subfolders
def listdir_nohidden(p):
    return glob.glob(os.path.join(p, '*'))


# list of file names in d
def get_file_names(d):
    all_files = os.listdir(d)
    if '.DS_Store' in all_files:
        all_files.remove('.DS_Store')
    return all_files


def clean_doc(doc):
    doc = doc[5:-2]
    paragraph = ' '.join(doc)
    return re.sub('\s+',' ', paragraph)


def clean_summary(summary):
    summary = summary[5:-2][0]
    summary = re.sub('\<.+?\>', '', summary)
    summary = summary[3:]
    return re.sub('\s+',' ', summary)


# read all documents/summaries and clean
def read_files(p):
    dirs = listdir_nohidden(p)
    file_paths = []
    for dir in dirs:
        file_names = get_file_names(dir)
        file_paths.extend([dir + '/' + file_name for file_name in file_names])
    files = []
    for file_path in file_paths:
        f = open(file_path, 'r')
        text = f.readlines()
        f.close()
        #cleaning
        if 'test_docs' in p:
            text = clean_doc(text)
        if 'gold_summaries' in p:
            text = clean_summary(text)
        files.append(text)
    return files


#write a function to calculate the KAPPA coeffieicents between each pair of human annotators for each document


# implement PAGE RANK like algorithm to rank the sentences in each document and return the one with highest score as 
# the summary of the document
# if that sounds scary, do the following:
# find the words in a document 
# define normalized-tf
# for each sentence:
# calculate a score by adding the normalized tf-score of the words present in the document
# for each sentence divide the total sentence score by the number of words (normalize as longer sentences might get more scores)
# return the highest ranked sentence as an extractive summary
def get_sentence(text):
    '''
    get a list of sentences
    return the most important sentence
    '''
    sentences = nltk.tokenize.sent_tokenize(text)
    words = dict()
    words_in_sents = list()
    for s in sentences:
        words_in_sent = list()
        tokens_raw = nltk.word_tokenize(s)
        for t in tokens_raw:
            if t.isalpha() and len(t) > 1:
                w = t.lower()
                words_in_sent.append(w)
                if w in words.keys():
                    words[w] += 1.0
                else:
                    words[w] = 1.0
        words_in_sents.append(words_in_sent)
    num_words = sum(words.values())
    for word in words:
        words[word] /= num_words
    s_score = list()
    for s in words_in_sents:
        score = 0.0
        for w in s:
            score += words[w]
        score /= len(s)
        s_score.append(score)
    print(s_score)
    m_index = s_score.index(max(s_score))
    return sentences[m_index]


#compare the similarity between your summary with each human-annotator's summary and return an average
#use jaccards simialrity
def jaccard(a, b):
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def clean(text):
    words = []
    tokens_raw = nltk.word_tokenize(text)
    for t in tokens_raw:
        if t.isalpha() and len(t) > 1:
            w = t.lower()
            words.append(w)
    return words


# my_summary: a string, result returned by get_sentence
# gold_summary: a string, corresponding gold summary sentence
def similarity_score(my_summary, gold_summary):
    return jaccard(clean(my_summary), clean(gold_summary))


# my summaries: a list of strings, where each string is a summary sentence returned by get_sentence
# gold_summaries: a list of strings, where each string is a gold summary sentence
def evaluate_model(my_summaries, gold_summaries):
    accum_jaccard = 0
    num_of_doc = len(my_summaries)
    num_of_summary = len(gold_summaries)
    for i in range(num_of_doc):
        accum_jaccard += similarity_score(my_summaries[i], gold_summaries[i])
    for i in range(num_of_doc):
        accum_jaccard += similarity_score(my_summaries[i], gold_summaries[num_of_doc+i])
    for i in range(num_of_doc):
        accum_jaccard += similarity_score(my_summaries[i], gold_summaries[2*num_of_doc+i])
    for i in range(num_of_doc):
        accum_jaccard += similarity_score(my_summaries[i], gold_summaries[3*num_of_doc+i])
    return accum_jaccard/num_of_summary

docs = read_files(path)
gold_summaries = read_files(gspath)
my_summaries = [get_sentence(doc) for doc in docs]
print(evaluate_model(my_summaries, gold_summaries))