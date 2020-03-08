
import csv
import copy
from sklearn.cluster import KMeans
from sklearn import svm
import random
import numpy as np

# attributes = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

TEST_PERCENT = 0.20
LABEL = 'num'
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

# Scales all data to [0,1]
def standardize_data(input_data):
    normal_data = copy.copy(input_data)
    for key in normal_data:
        if(key != LABEL):
            max_data = float(max(normal_data[key]))
            for i in range(len(normal_data[key])):
                normal_data[key][i] = float(normal_data[key][i]) / max_data
    return normal_data

def test_accuracy(correct_labels, generated_labels):
    num_labels = max(generated_labels) + 1
    correct = [0] * num_labels  
    for label_offset in range(num_labels):
        max_of_labels = [0] * num_labels
        for label in range(1, num_labels + 1):
            for j in range(len(correct_labels)):  # for each label of the same
                if(correct_labels[j] == label):   # value
                    if((correct_labels[j] + label_offset) % num_labels == generated_labels[j]):
                        max_of_labels[label - 1] += 1
            if max_of_labels[label - 1] > correct[label - 1]:
                correct[label - 1] = max_of_labels[label - 1]
    return sum(correct)/len(correct_labels)

def convert_array(in_dictionary):
    dictionary = copy.copy(in_dictionary)
    dictionary.pop(LABEL)
    arr = []
    i = 0
    for key in dictionary.keys():
        arr.append([])
        for data in dictionary[key]:
            arr[i].append(data)
        i += 1
    print(arr)
    return arr

input_data, attributes = get_data('ClevelandData.csv')

normal_data = standardize_data(input_data)

# Split Data
training_data, test_data = data_split(normal_data, TEST_PERCENT)
# Define K-Means
kmeans = KMeans(n_clusters = int(max(training_data[LABEL])), random_state = 0)

# K-Means
attributes_without_label = copy.copy(attributes)
attributes_without_label.pop(LABEL)
kmeans.fit(np.array([training_data[key] for key in attributes_without_label]).T)
training_labels = kmeans.labels_
