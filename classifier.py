## Imports

import csv, numpy, scipy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

## Functions

def load_data(path, include_labels=True):
    ''' Accepts a two column csv file with text and label data and returns a Pandas DataFrame (training) or list (testing) '''
    rows = []
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        line_number = 0
        for row in csvreader:
            line_number += 1
            if line_number > 1:
                if include_labels:
                    lines = {'text':row[0], 'class':row[1]}
                    rows.append(lines)
                else:
                    rows.append(row[0])

    if include_labels:
        return DataFrame(rows)
    else:
        return rows

def shuffle_data(data):
    ''' Accepts a DataFrame of training data and shuffles the rows '''
    return data.reindex(numpy.random.permutation(data.index))

def extract_features(data):
    ''' Accepts a DataFrame and extracts feature data - in this case words and their positions '''
    count_vectorizer = CountVectorizer(stop_words='english')
    feature_counts = count_vectorizer.fit_transform(data['text'].values)
    #print(count_vectorizer.vocabulary_)
    return feature_counts

def create_classifier(data, feature_counts):
    ''' Accepts an extracted CountVectorizer of feature counts and creates a trained NaiveBayes classifier to use for prediction'''
    classifier = MultinomialNB()
    targets = data['class'].values
    classifier.fit(feature_counts, targets)

    return classifier


## Testing



def test_load_data():
    training_data = [{'text':'UBER | Actue | Dizziness',
            'class':'Urgent'},
            {'text':'Discussing - has concerns and wants to talk about her fall ',
            'class':'Not urgent'}]
    testing_data = ['PC: Rash FU ','Acute | Chronic Cough']

    # print('\n')
    # print('DataFrame of training data:')
    # print(DataFrame(training_data))

    # print('\n')
    # print('Load training data from file:')
    # print(load_data('data/test_training_data.csv', include_labels=True))
    # print('\n')

    # print(load_data('data/test_testing_data', include_labels=False))
    # print('\n')
    # print(testing_data)
    # print('\n')

    assert DataFrame(training_data).equals(load_data('data/test_training_data.csv', include_labels=True))

    assert load_data('data/test_testing_data.csv', include_labels=False) == testing_data

# Tests below here are placeholders; come back and add real tests (besides printing) as a refactor step

def test_shuffle_data(data):
    ''' This really doesn't need to be tested, can write a test later if needed '''
    #print(shuffle_data(data))
    pass

def test_extract_features(data):
    features = extract_features(data)
    #print(features)
    #print(type(features))

def test_create_classifier(data, feature_counts):
    classifier = create_classifier(data, feature_counts)
    print(classifier)



## Run

data = load_data('data/TrainingData_925_to_1022.csv', True)
data = shuffle_data(data)
feature_counts = extract_features(data)
classifier = create_classifier(data, feature_counts)
examples = ["Sick visit - cough", "F/u on blood pressure"]
count_vectorizer = CountVectorizer(stop_words='english')
example_counts = count_vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)