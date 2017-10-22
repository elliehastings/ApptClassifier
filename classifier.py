## Imports

import csv
from pandas import DataFrame

## Functions

def load_training_data(path, include_labels=True):
    ''' Accepts a two column csv file with text and label data and returns a Pandas DataFrame '''
    rows = []
    with open('data/test_testing_data.csv', 'r') as csvfile:
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

def test_load_training_data():
    labeled_rows = [{'text':'UBER | Actue | Dizziness',
            'class':'Urgent'},
            {'text':'Discussing - has concerns and wants to talk about her fall ',
            'class':'Not urgent'}]
    unlabeled_rows = ['UBER | Actue | Dizziness','Discussing - has concerns and wants to talk about her fall ']
    assert DataFrame(labeled_rows).equals(load_training_data('data/test_testing_data.csv', include_labels=True))


def test_load_testing_data():
    pass

## Run

test_load_training_data()