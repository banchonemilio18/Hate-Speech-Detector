## For cleaning the csv file into easier to read files


import re
import csv

input_file = "HateSpeechDataset.csv"
tweets_file = "tweets.csv"
labels_file = "labels.csv"

## Found how to remove emojis from https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

def clean(tweet):
    tweet = emoji_pattern.sub('', tweet)
    tweet = re.sub(r"[~!@#$%^&*()_+-=`{}[];':<>,./]", '',  tweet)
    return tweet

tweets = []
labels = []

with open(input_file , newline= '', encoding='utf-8',) as file:
    reader = csv.reader(file)
    next(reader)

    for row in reader:
        if len(row) < 2: ##skip if bad line
            continue

    tweet = row[0].strip()
    label = row[1].strip()

    cleaned = clean(tweet)
    tweets.append(cleaned)
    labels.append(label)

#print("I am here")

with open(tweets_file, 'w', encoding='utf-8', newline='') as tweetsfile, open(labels_file, 'w', encoding='utf-8', newline='') as labelsfile:
    tweet_writer = csv.writer(tweetsfile)
    label_writer = csv.writer(labelsfile)

    for i in range(len(tweets)):
        tweet_writer.writerow([tweets[i]])
        label_writer.writerow([labels[i]])

print("Done")