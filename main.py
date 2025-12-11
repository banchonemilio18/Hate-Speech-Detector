import numpy as np
import pandas as pd # organize text and labels into a table (DataFrame)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample #randomly pick a smaller or larger group from a list

# ------- MAIN --------
# ask user to input text 
# call other Files and run instead of test but with input
# print the percent chance that it is hate speech

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

# Combine all training text and labels into a table for easier manipulation
df = pd.DataFrame({'text': X_train, 'label': Y_train})

# Separate classes
df_hate = df[df['label'] == 1]
df_nonhate = df[df['label'] == 0]

# Downsample non-hate (randomly select the same number of non-hate texts as hate texts to balance the dataset)
df_nonhate_downsampled = resample(df_nonhate, replace=False, n_samples=len(df_hate), random_state=42)

# Combine the hate and downsampled non-hate texts into one balanced dataset
df_balanced = pd.concat([df_hate, df_nonhate_downsampled])
X_train_balanced = df_balanced['text'].tolist()
Y_train_balanced = df_balanced['label'].tolist()

# Continue with TF-IDF and model
tfidf = TfidfVectorizer() # makes vectors out of lists
X_train_tfidf = tfidf.fit_transform(X_train_balanced)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=20)
model.fit(X_train_tfidf, Y_train_balanced)

txt = input("Enter text: ")
vector = tfidf.transform([txt])
pred = model.predict_proba(vector)[0]

print(f"{pred[1]:.2%} hate speech and", f"{pred[0]:.2%} not hate speech")