# Classifying a review as positive or negative by natural language processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# Importing the dataset
dataset = pd.read_csv('Reviews_of_Restaurants.tsv', delimiter = '\t', quoting = 3)

# Getting the text ready
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ready_reviews = []
for i in range(0, 1000):
    #takes the review and only keeps alphabets 
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #Coverts review into lowercase
    review = review.lower()
    #splits review as a list of words
    review = review.split()
    #Takes the root word and also neglect the words which will not be useful i.e. the,a etc
    Stemmer = PorterStemmer()
    review = [Stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #Converts the list into a string and append it to list of ready reviews
    review = ' '.join(review)
    ready_reviews.append(review)

#Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer(max_features = 800)
X = count_vec.fit_transform(ready_reviews).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Get accuracy
accuracy_percentage = accuracy_score(y_test, y_pred)*100
print("Accuracy of the model:" + str(accuracy_percentage))
