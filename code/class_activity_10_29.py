
# coding: utf-8

# In[1]:


import os
import glob
import re
import nltk
path = '../summarization_dataset_duc_2004/test_docs'
gspath = '../summarization_dataset_duc_2004/gold_summaries'


# In[2]:


def listdir_nohidden(p):
    return glob.glob(os.path.join(p, '*'))


# In[3]:


all_dirs = listdir_nohidden(path)
print(all_dirs)


# In[4]:


def get_file_names(d):
    all_files = os.listdir(d)
    #lst = all_files.split('\n')
    if '.DS_Store' in all_files:
        all_files.remove('.DS_Store')
    return all_files

some_files = get_file_names(all_dirs[0])
print(some_files)


# In[5]:


#get corresponding human-written summary files
gs_dirs = listdir_nohidden(gspath)
print(gs_dirs)

cor_summary_dir = os.path.split(all_dirs[0])[-1]
print(cor_summary_dir)
some_standard_summary_output = get_file_names(gspath+'/'+cor_summary_dir)
print(some_standard_summary_output)


# In[6]:


#read documents
file_0 = some_files[0]
f = open(all_dirs[0] + '/' + file_0, 'r')
text = f.readlines()
f.close()
print(text)
print(len(text))


# In[7]:


#cleaning
text = text[:-2]
print(text)
text = text[5:]
print(len(text))
print(text)


# In[8]:


#read human summary filenames for file_0
standard_summaries_file_0 = []
for fname in some_standard_summary_output:
    if file_0 in fname:
        #print(fname)
        standard_summaries_file_0.append(fname)
print(standard_summaries_file_0)


# In[9]:


#read human summaries
for f in standard_summaries_file_0:
    #print(f)
    if file_0 in f:
        print(file_0)
        print(f)
        p = open(gspath+'/'+ cor_summary_dir +'/'+ f, 'r')
        sl = p.readlines()
        p.close()
        sl = sl[5:-2]
        only_summary = sl[0]
        only_summary = re.sub('\<.+?\>', '', only_summary)
        only_summary = only_summary[3:]
        print(only_summary)



# In[10]:


#write a function to calculate the KAPPA coeffieicents between each pair of human annotators for each document


# In[11]:


# implement PAGE RANK like algorithm to rank the sentences in each document and return the one with highest score as 
#the summary of the document
# if that sounds scary, do the following:
# find the words in a document 
# define normalized-tf
# for each sentence:
#  calculate a score by adding the normalized tf-score of the words present in the document
# for each sentence divide the total sentence score by the number of words (normalize as longer sentences might get more scores)
#return the highest ranked sentence as an extractive summary

def get_sentence(sentences):
    '''
    get a list of sentences
    return the most important sentence
    '''
    words=dict()
    sents=list()
    for s in sentences:
        sent=list()
        tokens_raw = nltk.word_tokenize(s)
        for t in tokens_raw:
            if t.isalpha() and len(t)>1:
                w=t.lower()
                sent.append(w)
                if w in words.keys():
                    words[w]+=1.0
                else:
                    words[w]=1.0
        sents.append(sent)
    s_score=list()
    for s in sents:
        score=0.0
        for w in s:
            score+=words[w]
        score/=len(s)
        s_score.append(score)
    print(s_score)
    m=0.0
    m_index=0
    for i in range(len(s_score)):
        s=s_score[i]
        if s>m:
            m=s
            m_index=i
    return sentences[m_index]


print(text)
sent=get_sentence(text)
print(sent)

# In[12]:


#compare the similarity between your summary with each human-annotator's summary and return an average
#use jaccards simialrity 

