## Imports

import csv, numpy, scipy
from pandas import DataFrame, Series
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
        ('vectorizer',  CountVectorizer(stop_words='english', ngram_range=(1,4))),
        ('classifier',  MultinomialNB()) ])
    pipeline.fit(data['text'].values, data['class'].values)
    return pipeline.predict(examples) # ['label1','label2']

def write_predictions_to_file(y_examples, predicted_results, target_file_name):
    ''' Accepts an array of examples to classify and the predicted results and writes them to a csv file '''

    y_examples_array = numpy.array(y_examples, dtype=Series)

    with open(target_file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Text','Class'])
        for i in range(0, len(y_examples_array)):
            writer.writerow([y_examples_array[i], predicted_results[i]])


## Testing

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

def test_shuffle_data():
    test_dataframe = DataFrame([i for i in range(1,101)])
    shuffled_test_dataframe = shuffle_data(test_dataframe)
    try:
        shuffled_test_dataframe.iloc[0:5] != test_dataframe.iloc[0:5]
        raise AssertionError('Shuffled DataFrame equal to source DataFrame')
    except ValueError:
        pass


####### Tests ######

test_load_data()
test_shuffle_data()


####### Run #######

training_data = shuffle_data(load_data('data/TrainingData_925_to_1022.csv'))
testing_data = shuffle_data(load_data('data/TestingData_925_to_1022.csv'))
y_examples, y_results = testing_data['text'], testing_data['class']
predicted_results = create_and_classify_with_pipeline(training_data, y_examples)
f1_score = f1_score(y_results, predicted_results, pos_label='Urgent')
print("F1 Score is: {}".format(f1_score))
write_predictions_to_file(y_examples, predicted_results, 'data/output_predicted_results.csv')
print("Wrote predictions to file")