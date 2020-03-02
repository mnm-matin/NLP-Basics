"""
Foundations of Natural Language Processing
Assignment 1: Corpora Analysis and Language Identification

Please complete functions, based on their doc_string description
and instructions of the assignment. 

To test your code run:

```
[hostname]s1234567 python3 s1234567.py
```

Before submittion executed your code with ``--answers`` flag
```
[hostname]s1234567 python3 s1234567.py --answers
```
include generated answers.py file and generated plots to
your submission.

Best of Luck!
"""

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk.corpus import inaugural, brown  # import corpora
from nltk.corpus import stopwords  # stopword list

# Import the Twitter corpus and LgramModel
try:
    from nltk_model import *  # See the README inside the nltk_model folder for more information
    from twitter import *
except ImportError:
    from .nltk_model import *  # Compatibility depending on how this script was run
    from .twitter import *


# Override default plotting function in matplotlib so that your plots would be saved during the execution
from matplotlib import pylab, pyplot
plot_enum = 0

def my_show(**kwargs):
    global plot_enum
    plot_enum += 1
    return pylab.savefig('plot_{}.png'.format(plot_enum))

pylab.show=my_show
pyplot.show=my_show

twitter_file_ids = "20100128.txt"
assert twitter_file_ids in xtwc.fileids()

# Helper function to get tokens of corpus containing multiple files
def get_corpus_tokens(corpus, list_of_files):
    '''Get the tokens from (part of) a corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :type list_of_files: list(str) or str
    :param list_of_files: file(s) to read from
    :rtype: list(str)
    :return: the tokenised contents of the files'''

    # Get a list of all tokens in the corpus
    corpus_tokens = corpus.words(list_of_files)
    # Return the list of corpus tokens
    return corpus_tokens

# =======================================
# Section A: Corpora Analysis [45 marks]
# =======================================

# Question 1 [5 marks]
def avg_type_length(corpus, list_of_files):
    '''
    Compute the average word type length from (part of) a corpus
    specified in list_of_files

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :type list_of_files: list(str) or str
    :param list_of_files: file(s) to read from
    :rtype: float
    :return: the average word type length over all the files
    '''

    raise NotImplementedError # remove when you finish defining this function

    # Get a list of all tokens in the corpus
    tokens = ...

    # Construct a list that contains the token lengths for each DISTINCT token in the document
    distinct_token_lengths = ...

    # Return the average type length of the document
    return ...

# Question 2 [5 marks]
def open_question_1():
    '''
    Question: Why might the average type length be greater for twitter data?

    :rtype: str
    :return: your answer'''

    return inspect.cleandoc("""\
        ...""")[0:500]

# Question 3 [10 marks]
def plot_frequency(tokens, topK=50):
    '''
    Tabulate and plot the top x most frequently used word types
    and their counts from the specified list of tokens

    :type tokens: list(str) 
    :param tokens: List of tokens
    :type topK: int
    :param number of top tokens to plot and return as a result
    :rtype: list(tuple(string,int))
    :return: top K word types and their counts from the files
    '''
    raise NotImplementedError # remove when you finish defining this function

    # Construct a frequency distribution over the lowercased tokens in the document
    fd_doc_types = ...

    # Find the top topK most frequently used types in the document
    top_types = ...

    # Produce a plot showing the top topK types and their frequencies

    # Return the top topK most frequently used types
    return ...

# Question 4 [15 marks]
def clean_data(corpus_tokens):
    '''
    Clean a list of corpus tokens

    :type corpus_tokens: list(str)
    :param corpus_tokens: (lowercased) corpus tokens
    :rtype: list(str)
    :return: cleaned list of corpus tokens
    '''
    raise NotImplementedError # remove when you finish defining this function

    stops = list(stopwords.words("english"))

    # A token is 'clean' if it's alphanumeric and NOT in the list of stopwords

    # Return a cleaned list of corpus tokens
    return ...

# Question 5 [10 marks]
def open_question_2():
    '''
    Problem: noise in Twitter data

    :rtype: str
    :return: your answer []
    '''
    return inspect.cleandoc("""\
        ... """)[0:500]

# ==============================================
# Section B: Language Identification [45 marks]
# ==============================================

# Question 6 [15 marks]
def train_LM(corpus):
    '''
    Build a bigram letter language model using LgramModel
    based on the all-alpha subset the entire corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :rtype: LgramModel
    :return: A padded letter bigram model based on nltk.model.NgramModel
    '''
    raise NotImplementedError # remove when you finish defining this function

    # subset the corpus to only include all-alpha tokens
    corpus_tokens= ...

    # Return a smoothed padded bigram letter language model
    return ...

# Question 7 [15 marks]
def tweet_ent(file_name,bigram_model):
    '''
    Using a character bigram model, compute sentence entropies
    for a subset of the tweet corpus, removing all non-alpha tokens and
    tweets with less than 5 all-alpha tokens

    :type file_name: str
    :param file_name: twitter file to process
    :rtype: list(tuple(float,list(str)))
    :return: ordered list of average entropies and tweets
    '''

    raise NotImplementedError # remove when you finish defining this function

    # Clean up the tweet corpus to remove all non-alpha 
    # # tokens and tweets with less than 5 (remaining) tokens
    list_of_tweets = xtwc.sents(file_name)
    cleaned_list_of_tweets = ... 

    # Construct a list of tuples of the form: (entropy,tweet)
    #  for each tweet in the cleaned corpus, where entropy is the
    #  average word for the tweet, and return the list of
    #  (entropy,tweet) tuples sorted by entropy

    return ...
# Question 8 [10 marks]
def open_question_3():
    '''Question: What differentiates the beginning and end of the list
       of tweets and their entropies?

    :rtype: str
    :return: your answer [500 chars max]'''

    return inspect.cleandoc("""\
    ...""")[0:500]


# Question 9 [15 marks]
def tweet_filter(list_of_tweets_and_entropies):
    '''
    Compute entropy mean, standard deviation and using them,
    likely non-English tweets in the all-ascii subset of list 
    of tweetsand their biletter entropies

    :type list_of_tweets_and_entropies: list(tuple(float,list(str)))
    :param list_of_tweets_and_entropies: tweets and their
                                    english (brown) average biletter entropy
    :rtype: tuple(float, float, list(tuple(float,list(str)))
    :return: mean, standard deviation, ascii tweets and entropies,
             not-English tweets and entropies
    '''

    raise NotImplementedError # remove when you finish defining this function

    # Find the "ascii" tweets - those in the lowest-entropy 90%
    #  of list_of_tweets_and_entropies
    list_of_ascii_tweets_and_entropies = ...

    # Extract a list of just the entropy values
    list_of_entropies = ...

    # Compute the mean of entropy values for "ascii" tweets
    mean = ...

    # Compute their standard deviation
    standard_deviation = ...

    # Get a list of "probably not English" tweets, that is
    #  "ascii" tweets with an entropy greater than (mean + std_dev))
    threshold = ...
    list_of_not_English_tweets_and_entropies= ...

    # Return mean, standard_deviation,
    #  list_of_ascii_tweets_and_entropies,
    #  list_of_not_English_tweets_and_entropies
    return ...

# Utility function
def ppEandT(eAndTs):
    '''
    Pretty print a list of entropy-tweet pairs

    :type eAndTs: list(tuple(float,list(str)))
    :param eAndTs: entropies and tweets
    :return: None
    '''

    for entropy,tweet in eAndTs:
        print("{:.3f} [{}]".format(entropy,", ".join(tweet)))

"""
Format the output of your submission for both development and automarking. 
DO NOT MODIFY THIS PART !
""" 
def answers():

    # Global variables for answers that will be used by automarker
    global avg_inaugural, avg_twitter, top_types_inaugural, top_types_twitter
    global tokens_inaugural, tokens_clean_inaugural, total_tokens_clean_inaugural
    global fst100_clean_inaugural, top_types_clean_inaugural
    global tokens_twitter, tokens_clean_twitter, total_tokens_clean_twitter
    global fst100_clean_twitter, top_types_clean_twitter, lm
    global best10_ents, worst10_ents, mean, std, best10_ascci_ents, worst10_ascci_ents
    global best10_non_eng_ents, worst10_non_eng_ents
    global answer_open_question_1, answer_open_question_2, answer_open_question_3

    # Question 1
    print("*** Question 1 ***")
    avg_inaugural = avg_type_length(inaugural, inaugural.fileids())
    avg_twitter = avg_type_length(xtwc, twitter_file_ids)
    print("Average token length for Inaugural corpus: {:.2f}".format(avg_inaugural))
    print("Average token length for Twitter corpus: {:.2f}".format(avg_twitter))
    
    # Question 2
    print("*** Question 2 ***")
    answer_open_question_1 = open_question_1()
    print(answer_open_question_1)

    # Question 3
    print("*** Question 3 ***")
    print("Most common 50 types for the Inaugural corpus:")
    tokens_inaugural = get_corpus_tokens(inaugural,inaugural.fileids())
    top_types_inaugural = plot_frequency(tokens_inaugural,50)
    print(top_types_inaugural)
    print("Most common 50 types for the Twitter corpus:")
    tokens_twitter = get_corpus_tokens(xtwc, twitter_file_ids)
    top_types_twitter = plot_frequency(tokens_twitter,50)
    print(top_types_twitter)

    # Question 4
    print("*** Question 4 ***")
    tokens_inaugural = get_corpus_tokens(inaugural,inaugural.fileids())
    tokens_clean_inaugural = clean_data(tokens_inaugural)
    total_tokens_inaugural = len(tokens_inaugural)
    total_tokens_clean_inaugural = len(tokens_clean_inaugural)
    print("Inaugural Corpus:")
    print("Number of tokens in original corpus: {}".format(total_tokens_inaugural))
    print("Number of tokens in cleaned corpus: {}".format(total_tokens_clean_inaugural))
    print("First 100 tokens in cleaned corpus:")
    fst100_clean_inaugural = tokens_clean_inaugural[:100]
    print(fst100_clean_inaugural)
    print("Most common 50 types for the cleaned corpus:")
    top_types_clean_inaugural = plot_frequency(tokens_clean_inaugural,50)
    print(top_types_clean_inaugural)

    print('-----------------------')
    
    tokens_twitter = get_corpus_tokens(xtwc,twitter_file_ids)
    tokens_clean_twitter = clean_data(tokens_twitter)
    total_tokens_twitter = len(tokens_twitter)
    total_tokens_clean_twitter = len(tokens_clean_twitter)
    print("Twitter Corpus:")
    print("Number of tokens in original corpus: {}".format(total_tokens_twitter))
    print("Number of tokens in cleaned corpus: {}".format(total_tokens_clean_twitter))
    print("First 100 tokens in cleaned corpus:")
    fst100_clean_twitter = tokens_clean_twitter[:100]
    print(fst100_clean_twitter)
    print("Most common 50 types for the cleaned corpus:")
    top_types_clean_twitter = plot_frequency(tokens_clean_twitter,50)
    print(top_types_clean_twitter)
    # Question 5
    print("*** Question 5 ***")
    answer_open_question_2 = open_question_2()
    print(answer_open_question_2)
    
    # Question 6
    print("*** Question 6 ***")
    print('Building brown bigram letter model ... ')
    lm = train_LM(brown)
    print('Letter model buid')

    # Question 7
    print("*** Question 7 ***")
    ents = tweet_ent(twitter_file_ids, lm)
    print("Best 10 english entropies:")
    best10_ents = ents[:10]
    ppEandT(best10_ents)
    print("Worst 10 english entropies:")
    worst10_ents = ents[-10:]
    ppEandT(worst10_ents)

    # Question 8
    print("*** Question 8 ***")
    answer_open_question_3 = open_question_3()
    print(answer_open_question_3)

    # Question 9
    print("*** Question 9 ***")
    mean, std, ascci_ents, non_eng_ents = tweet_filter(ents)
    print('Mean: {}'.format(mean))
    print('Standard Deviation: {}'.format(std))
    print('ASCII tweets ')
    print("Best 10 English entropies:")
    best10_ascci_ents = ascci_ents[:10]
    ppEandT(best10_ascci_ents)
    print("Worst 10 English entropies:")
    worst10_ascci_ents = ascci_ents[-10:]
    ppEandT(worst10_ascci_ents)
    print('--------')
    print('Non-English tweets ')
    print("Best 10 English entropies:")
    best10_non_eng_ents = non_eng_ents[:10]
    ppEandT(best10_non_eng_ents)
    print("Worst 10 English entropies:")
    worst10_non_eng_ents = non_eng_ents[-10:]
    ppEandT(worst10_non_eng_ents)

if __name__ == "__main__":

    answers()

    if len(sys.argv)>1 and sys.argv[1] == '--answers':
        ans=['avg_inaugural', 'avg_twitter',
             ('answer_open_question_1', repr(answer_open_question_1)),
             'top_types_inaugural', 'top_types_twitter',
             'total_tokens_clean_inaugural',
             'fst100_clean_inaugural', 'top_types_clean_inaugural',
             'total_tokens_clean_twitter',
             'fst100_clean_twitter','top_types_clean_twitter',
             ('answer_open_question_2',repr(answer_open_question_2)),
             ('lm_stats',[lm._N,
                         lm.prob('h','t'),
                         lm.prob('u','q'),
                         lm.prob('z','q'),
                         lm.prob('j',('<s>',),True),
                         lm.prob('</s>','e',True)]),
             'best10_ents','worst10_ents',
             ('answer_open_question_3', repr(answer_open_question_3)),
             'mean','std',
             'best10_ascci_ents', 'worst10_ascci_ents',
             'best10_non_eng_ents', 'worst10_non_eng_ents']

        ansd={}
        for an in ans:
            if type(an) is tuple:
                ansd[an[0]]=an[1]
            else:
                ansd[an]=eval(an)
        # dump for automarker
        with open('answers.py',"w") as f:
            for aname,aval in ansd.items():
                print("{}={}".format(aname,aval),file=f)
