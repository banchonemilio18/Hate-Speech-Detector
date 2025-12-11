# Hate Speech Detector Project

This project focuses on building and comparing two machine-learning models to classify text as either hate speech or non–hate speech. The models are trained using the Hate Speech Detection Curated Dataset, created by Wendyellé A. Alban Nyantudre and hosted on Kaggle.

We implemented two different approaches to evaluate how various machine-learning algorithms perform on hate speech classification. Our first method is a Naive Bayes classifier, which achieved an accuracy of 95% on the dataset. This model is lightweight, fast to train, and works particularly well for text-based classification tasks due to its ability to handle high-dimensional data. The second method uses a Random Forest classifier from the scikit-learn library. This approach reached an accuracy of 82% but required significantly more computation time because the dataset is large and Random Forest models are more resource-intensive. Both models rely on the same cleaned dataset produced by our cleandata.py script, and both feed into the interactive program found in main.py.

This repository includes several Python files that support this workflow. The cleandata.py script prepares the raw dataset by cleaning text, removing noise, and formatting it so it can be used for training. The NaiveBayes.py file trains and tests the Naive Bayes model, while RandomizedForest.py handles the Random Forest approach and outputs evaluation metrics. The main.py file provides a simple command-line experience where users can type in any sentence or phrase and receive a prediction indicating whether the text is likely to be hate speech.

How to run each of these programs?
Users can type python NaiveBayes.py, python RandomizedForest.py, or python main.py in the terminal. The Random Forest model may take longer to execute due to its complexity and the size of the dataset.

The dataset used in this project is publicly available on Kaggle and can be accessed here:
Dataset Page: https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset

(You must be logged into Kaggle to download the CSV file. Once downloaded, the CSV should be placed in the project directory so it can be processed and used by the scripts)
