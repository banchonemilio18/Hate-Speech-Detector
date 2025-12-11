import re
import sys
import os
import math#added this for log
from collections import defaultdict # added for error of missing words 

class hateDetector:
    """
    A hate detector implemented using Naive Bayes. 
    Be sure to use instance variables (i.e. self.foo) to store data, do not 
    declare these variables outside of methods, as that will make them 
    static class variables.
    """
    
    def __init__(self):
        self.hate_word_count = defaultdict(int)# how often word show up in hate messages
        self.non_hate_word_count = defaultdict(int)
        self.hate_count = 0 # number of hate messages
        self.non_hate_count = 0
        self.total_hate_words = 0 #log_p_word_given_hate
        self.total_non_hate_words = 0
        #pass
    
    def train(self, hate_messages, non_hate_messages):
        """
        Count the number of times each word occurs in each set of messages
        in order to precompute the values for the methods below
        """
        self.hate_count = len(hate_messages)
        self.non_hate_count = len(non_hate_messages)
        
        for message in hate_messages:
            for word in message:
                self.hate_word_count[word] +=1
        
        for message in non_hate_messages:
            for word in message:
                self.non_hate_word_count[word] +=1

        self.total_non_hate_words = sum(self.non_hate_word_count.values()) # gets total number of values from dictionary
        self.total_hate_words = sum(self.hate_word_count.values())
        #pass
    
    def log_p_hate(self):
        """
        returns log p(hate)
        this method should never return 0
        """
        return math.log(self.hate_count/ (self.hate_count + self.non_hate_count))
        #pass
    
    def log_p_non_hate(self):
        """
        returns log p(non_hate)
        this method should never return 0
        """
        return math.log(self.non_hate_count/ (self.hate_count + self.non_hate_count))
        #pass
    
    def log_p_word_given_hate(self, word):
        """
        returns log p(word | hate)
        this method should never return 0
        """ 
        # wordcount / total
        wordcount = self.hate_word_count[word] + 1
        total = self.total_hate_words + len(self.hate_word_count)
        return math.log(wordcount/total)

        #pass
    
    def log_p_word_given_non_hate(self, word):
        """
        returns log p(word | non_hate)
        this method should never return 0
        """
        # wordcount / total
        wordcount = self.non_hate_word_count[word] + 1
        total = self.total_non_hate_words + len(self.non_hate_word_count)
        return math.log(wordcount/total)

        #pass
    
    def is_hate(self, message):
        """
        Takes a list of words and returns True if the message is more likely
        hate than non_hate, and False otherwise
        """

        hate = self.log_p_hate()
        non_hate = self.log_p_non_hate()

        for word in message:
            hate += self.log_p_word_given_hate(word)
            non_hate += self.log_p_word_given_non_hate(word)
        
        if hate > non_hate:
            return True
        else:
            return False

        #pass
    
    def predict_all(self, messages):
        """
        Takes a list of test messages and calls predict on each
        Returns the result
        """
        predictions = []
        for message in messages:
            predictions.append(self.is_hate(message))
        return predictions

def load(fname, stop_words):
    """
    Load, tokenize and clean a text file
    Removes uninformative words called "stop words" from the text
    """
    with open(fname, encoding='utf-8', errors='ignore') as f:
        text = f.read().splitlines()

    tokens = []
    for doc in text:
        tok = re.split(r'[ \n\t.,:!\?\;\-\“\”\(\)]+', doc.lower())
        tok = [t for t in tok if t != '' and t not in stop_words]
        tokens.append(tok)

    return tokens

def score(pred, true):
    """
    Return the accuracy score for pred and true
    """
    return sum([1 if p == t else 0 for p,t in zip(pred, true)]) / len(pred)

# main method: don't edit
if __name__ == '__main__':
    with open(os.path.join('stop_words.txt'), encoding='latin1') as f:
        stop_words = set(f.read().split('\n'))

    hate = load(os.path.join('hate_training.txt'), stop_words)
    non_hate = load(os.path.join('nonhate_training.txt'), stop_words)

    model = hateDetector()
    model.train(hate, non_hate)

    test = load(os.path.join('test.txt'), stop_words)
    with open(os.path.join('labels.txt')) as f:
        labels = [a == 'hate' for a in f.read().split('\n')]
    predictions = model.predict_all(test)

    print("Score: ", score(predictions,labels)) #added for testing