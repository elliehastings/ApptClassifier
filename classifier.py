## Imports

import csv, numpy, scipy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score

## Functions

def load_data(path):
    ''' Accepts a two column csv file with text and label data and returns a Pandas DataFrame (training) or list (testing) '''
    rows = []
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        line_number = 0
        for row in csvreader:
            line_number += 1
            if line_number > 1:
                lines = {'text':row[0], 'class':row[1]}
                rows.append(lines)

    return DataFrame(rows)

def shuffle_data(data):
    ''' Accepts a DataFrame of training data and shuffles the rows '''
    return data.reindex(numpy.random.permutation(data.index))

def create_and_classify_with_pipeline(data, examples):
    '''Accepts training data, and testing data and returns predicted results'''
    pipeline = Pipeline([
        ('vectorizer',  CountVectorizer(stop_words='english')),
        ('classifier',  MultinomialNB()) ])
    pipeline.fit(data['text'].values, data['class'].values)
    return pipeline.predict(examples) # ['label1','label2']


## Testing - stopped after one function :)

def test_load_data():
    training_data = [{'text':'UBER | Actue | Dizziness',
            'class':'Urgent'},
            {'text':'Discussing - has concerns and wants to talk about her fall ',
            'class':'Not urgent'}]
    testing_data = [{'text':'PC: Rash FU ',
            'class':'Not urgent'},
            {'text':'Acute | Chronic Cough',
            'class':'Urgent'}]

    assert DataFrame(training_data).equals(load_data('data/test_training_data.csv'))

    assert DataFrame(testing_data).equals(load_data('data/test_testing_data.csv'))



####### Run #######

training_data = shuffle_data(load_data('data/TrainingData_925_to_1022.csv'))
testing_data = shuffle_data(load_data('data/TestingData_925_to_1022.csv'))
y_examples, y_results = testing_data['text'], testing_data['class']
predicted_results = create_and_classify_with_pipeline(training_data, y_examples)
f1_score = f1_score(y_results, predicted_results, pos_label='Urgent')
print("F1 Score is: {}".format(f1_score))

