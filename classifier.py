## Imports

import csv
from pandas import DataFrame

## Functions

def load_training_data(path):
    rows = []
    with open('data/test_data.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        line_number = 0
        for row in csvreader:
            line_number += 1
            if line_number > 1:
                line_dict = {'text':row[0], 'class':row[1]}
                rows.append(line_dict)
    return DataFrame(rows)

def load_testing_data(path):
    pass

## Testing

def test_load_training_data():
    rows = [{'text':'UBER | Actue | Dizziness',
            'class':'Urgent'},
            {'text':'Discussing - has concerns and wants to talk about her fall ',
            'class':'Not urgent'}]
    #print(load_training_data('data/test_data.csv'))
    #print(rows)
    assert DataFrame(rows).equals(load_training_data('data/test_data.csv'))

def test_load_testing_data():
    pass

## Run

test_load_training_data()