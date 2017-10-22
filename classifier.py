## Imports

import csv
from pandas import DataFrame

## Functions

def load_data(path, include_labels=True):
    ''' Accepts a two column csv file with text and label data and returns a Pandas DataFrame '''
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


def test_load_testing_data():
    pass

## Run

test_load_data()