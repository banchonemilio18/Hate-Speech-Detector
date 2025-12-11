import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def score(pred, true):
    return sum([1 if p == t else 0 for p,t in zip(pred, true)]) / len(pred)

with open('nonhate_training.txt' , 'r', encoding='utf-8') as f:
    nonhate_text = f.readlines()

with open('hate_training.txt' , 'r', encoding='utf-8') as f:
    hate_text = f.readlines()

with open('test.txt' , 'r', encoding='utf-8') as f:
    test_text = f.readlines()

with open('labels.txt' , 'r', encoding='utf-8') as f:
    labels_text = [int(line.strip()) for line in f]

# create one training list and make on label list for it
X_train = nonhate_text + hate_text
Y_train = [0] * len(nonhate_text) + [1] * len(hate_text)

#TF-IDF = Term Frequency-Inverse Document Frequency
#TFIDF makes important words have higher weight 
# tfidf to turn lists into vectors for sk learns training
tfidf = TfidfVectorizer() # makes vectors out of lists
X_train_tfidf = tfidf.fit_transform(X_train)#learn vocab then makes vector
test_tfidf = tfidf.transform(test_text)#turn into vector based on what was learned from 

model = RandomForestClassifier(n_estimators=10, random_state=20)
model.fit(X_train_tfidf, Y_train) # trains model by building decision trees 

pred = model.predict(test_tfidf)
print("Score: ", score(pred,labels_text)) #added for testing