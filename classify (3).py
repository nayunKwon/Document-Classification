# netid : nkwon23
# class : CS540

import os
import math


# These first two functions require os operations and so are completed for you
# Completed for you


def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        if os.path.isdir(directory+subdir):
            files = os.listdir(directory+subdir)
            for f in files:
                bow = create_bow(vocab, directory+subdir+f)
                dataset.append({'label': label, 'bow': bow})
    return dataset

# Completed for you


def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        if os.path.isdir(directory+subdir):
            files = os.listdir(directory+subdir)
            for f in files:
                with open(directory+subdir+f, 'r') as doc:
                    for word in doc:
                        word = word.strip()
                        if not word in vocab and len(word) > 0:
                            vocab[word] = 1
                        elif len(word) > 0:
                            vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

# The rest of the functions need modifications ------------------------------
# Needs modifications


def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """

    bow = {}
    with open(filepath, 'r') as d:
        for line in d:
            w = line.strip()
            if w in bow:  #if w is in the bow
                bow[w] += 1  
            elif w in vocab:  #if w is not in the bow, but in the vocab
                bow[w] = 1
            else:
                if None in bow:   # if OOV is in the bow
                    bow[None] += 1
                else:   # if OOV is not in the bow
                    bow[None] = 1

    return bow

# Needs modifications


def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1  # smoothing factor
    logprob = {}

    for l in label_list:
        count = 0
        for i in range(len(training_data)):
            if training_data[i].get('label') == l:
                count += 1
        logprob[l] = math.log((count + smooth) / (len(training_data)+2))

    return logprob


# Needs modifications


def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1  # smoothing factor
    word_prob = {}

    for i in vocab:
        word_prob[i] = 0  
    word_prob[None] = 0

    b = [bow['bow'] for bow in training_data if bow['label'] == label]  

    for bow in b:
        for w in bow:
            word_prob[w] += bow[w]

    t = sum(word_prob.values()) #total

    for word in word_prob:
        word_prob[word] = math.log((word_prob[word]+smooth) / (t+len(vocab)+1))

    return word_prob


##################################################################################
# Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)

    retval = {'vocabulary': vocab, 'log prior': prior(training_data, label_list),
          'log p(w|y=2016)': p_word_given_label(vocab, training_data, '2016'),
          'log p(w|y=2020)': p_word_given_label(vocab, training_data, '2020')}

    return retval

 

# Needs modifications


def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    vocab = model['vocabulary']
    cb = create_bow(vocab, filepath)

    lp_16 = cal_h(cb, model['log p(w|y=2016)'], model['log prior']['2016'])  # use helper func to calculate log probability of each year
    lp_20 = cal_h(cb, model['log p(w|y=2020)'], model['log prior']['2020'])
    
    if lp_16 > lp_20:
        retval['predicted y'] = '2016'
    else:
        retval['predicted y'] = '2020'

    retval['log p(y=2016|x)'] = lp_16
    retval['log p(y=2020|x)'] = lp_20

    return retval

# helper function to calculate
# log probailities of each year(label)
# 
def cal_h(bow, model, num):
    dic = {}
    lp = 0

    for w in bow: 
        dic[w] = bow[w]
    for w in dic:
        for w in range(dic[w]):
            lp += model[w]
    lp += num
    
    return lp
