
import csv
import copy
from sklearn.cluster import KMeans
from sklearn import svm
import random
import numpy as np

# attributes = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

TEST_PERCENT = 0.20

'''
class value:
    def __init__(self, data_arr, attributes):
        self.data = {}
        for i in range(len(attributes)):
            self.data[attributes[i]] = data_arr[i]

# In: name: file name  &  label: True if the data contains labels for data, and False if regular data
# Out: data: array that contains all the data
# Out: data_label: array of lables for data
def get_data(name):
    global DIM
    global K
    data = []
    attributes = []
    with open(name, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for entry in reader:
            DIM = len(entry)
            temp = []
            for value in entry:
                if(value == '?'):
                    temp.append(value)
                else:
                    try:
                        temp.append(float(value))
                    except:
                        attributes.append(value)
            if len(temp) != 0:
                data.append(temp)
    data_array(data, attributes)
    return data


def data_array(data, attributes):
    data_array = []
    for entry in data:
        p1 = value(entry, attributes)
        data_array.append(p1)
    print(data_array[0].data)
    return data_array
        
'''

def data_split(input_data, test_percent):
    training_data = copy.copy(input_data)
    test_data = {}
    for key in input_data:
        test_data[key] = []
    for i in range(int(len(input_data['num']) * test_percent)):
        index = random.randrange(len(training_data['num']))
        for key in input_data:
            test_data[key].append(training_data[key].pop(index)) 
    return training_data, test_data

# In: name: file name  &  label: True if the data contains labels for data, and False if regular data
# Out: data: array that contains all the data
# Out: data_label: array of lables for data
def get_data(name):
    global DIM
    global K
    data = {}
    attributes = []
    with open(name, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headings = next(reader)
        for i in range(len(headings)):
            data[headings[i]] = []
            attributes.append(headings[i])
        for entry in reader:
            DIM = len(entry)
            for i in range(len(entry)):
                if(entry[i] == '?'):
                    data[attributes[i]].append(entry[i])
                else:
                    data[attributes[i]].append(float(entry[i]))
    replace_with_median(data, attributes)
    return data, attributes

# Repalce all '?' with the median of the attribute
def replace_with_median(data, attributes):
    for title in attributes:
        for i in range(len(data[title])):
            middle = int(len(data[title])/2)
            if data[title][i] == '?':
                while (data[title][middle] == '?'):
                    middle += 1
                data[title][i] = data[title][middle]
def print_point(data, index):
    for attribute in attributes:
        print(attribute + ": " + str(data[attribute][0]))

input_data, attributes = get_data('ClevelandData.csv')

# Split Data
training_data, test_data = data_split(input_data, TEST_PERCENT)
# Define K-Means
kmeans = KMeans(n_clusters = 3, random_state = 0)

# K-Means
kmeans.fit(np.array([training_data[key] for key in attributes]).T)
training_labels = kmeans.labels_