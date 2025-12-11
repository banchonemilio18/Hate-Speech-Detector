# HateSpeechDetector
A Hate Speech Detector

We created a hate speech detector with two methods. The first method being a naive bayes approach which has 95% accuracy with the dataset we used. Our second approach has 82% accuracy and used sci-kit learn's randomized forest approach. The data set we used is Hate Speech Detection curated Dataset🤬 by Wendyellé A. Alban NYANTUDRE on kaggle.com. Our main.py file trains on our test data then asks the user to input some text and our model predicts the likely hood that the text is hate speech or non-hate speech. The cleandata.py method is how we cleaned the initial csv file.

To run NaiveBayes.py in the terminal use this prompt: python NaiveBayes.py
To run RandomizedForest.py in the terminal use this prompt: python RandomizedForest.py
To run main.py in the terminal use this prompt: python main.py

All three prompts may take a few moments to run becuase the dataset is so large specifically the RandomizedForest.