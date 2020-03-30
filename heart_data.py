
import csv
import copy
from sklearn.cluster import KMeans
from sklearn import svm
import random
import numpy as np

# attributes = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

random.seed(0)

TEST_PERCENT = 0.20
LABEL = 'num'
GROUP_CAP = 5           # Cap for the number of groups used to determine information gain
MAX_LABEL = 0             # Determined from code
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
            min_data = float(min(normal_data[key]))
            max_data = float(max(normal_data[key])) - min_data
            for i in range(len(normal_data[key])):
                normal_data[key][i] = (float(normal_data[key][i]) - min_data) / max_data
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

# Divides result into categories with the determining attributes (may want to change name later)
def entropy_groups(attribute, data):
    tree = []
    group = []
    over_cap = False
    for i in range(len(data[attribute])):
        val = data[attribute][i]
        if len(group) > GROUP_CAP:
            over_cap = True
            break
        elif val not in group:
            group.append(val)                           # Holds the number of features to check if over cap
            tree.append([])
            tree[group.index(val)].append(val)          # Appends the feature
        tree[group.index(val)].append(data[LABEL][i])   # Appends label associated to feature
    if over_cap:
        tree = []
        group_count = 1 / GROUP_CAP
        for i in range(GROUP_CAP):
            tree.append([])
            tree[i].append(group_count * (i + 1))       # Creates segmented dividers
        for i in range(len(data[attribute])):
            for j in tree:
                if data[attribute][i] <= j[0]:           # Determines where label belongs in dividers based on actual feature
                    j.append(data[LABEL][i])
                    break  
    print (tree)
    return tree

# Calculates the fraction of each label present in each attribute
# This was done as binary values, so needs to be repeated with each label and must be weighted
def binary_entropy(group):
    t_group = copy.deepcopy(group)
    num_label = []
    total_labels = []
    labels = [0] * MAX_LABEL        # Array to hold the total number of labels for each feature threshold
    loop = 0
    total = 0
    for feature in t_group:
        all_attr = 0
        num_label.append([0] * (MAX_LABEL + 2))  # [attribute name, label fraction, non-label fraction, weight of attribute]
        num_label[loop][0] = feature.pop(0)
        for i in feature:
            all_attr += 1
            num_label[loop][int(i + 1)] += 1
        num_label[loop].append(all_attr)
        total_labels.append(all_attr)
        if all_attr:
            for i in range(MAX_LABEL + 1):
                num_label[loop][int(i + 1)] /= all_attr
        total += all_attr
        loop += 1
    print(total_labels)
    for i in range(len(total_labels)):
        total_labels[i] /= total
    print(total_labels)
    for i in num_label:
        i[int(MAX_LABEL + 2)] /= total
    #print(num_label)
    print()
    return num_label, total_labels

# Calculates the entropy values for each attribute (treats them as binary values)
''' TODO: find out how to calculate/weigh with all the binary values to get nonbinary answer 
          also clean up the formatting of the output 
          also deal with nan values
          also deal with why exang returns neg info gain'''
def entropy(attribute, data):
    information_gain = 0
    system_entropy = 0
    groups = entropy_groups(attribute, data)
    final = []
    dec_groups, total_labels = binary_entropy(groups)
    for i in total_labels:
        if i:
            system_entropy -= (i * np.log2(i))
    print('sys entropy: ', system_entropy)
    for decimal in dec_groups:
        val = 0
        print(decimal.pop(0), '---------------------')
        weight = decimal.pop(len(decimal) - 1)
        print(weight)
        print(decimal)
        for j in decimal:   # Calculates entropy for each label value for every attribute
            if j != 0:
                val -= (j * np.log2(j))
        #val = -val
        print("val:", str(val))
        val *= weight       # Applies the weight to the entropy to find the information gain
        print("weighted val:", str(val))
        print()
        '''if val != val:          # Just makes nan values 0 for the time being, as I brainstorm how to solve the issue
            val = 0'''
        final.append(val)       # final : [find label values][find feature values][0 = actual feature value, 1 = entropy value]
    print('final: ',final)
    information_gain = (system_entropy - sum(final))
    return information_gain

'''def info_gain(attribute, data):
    for label_val in data:
        i_g = 0
        for feat_val in label_val:'''
            

input_data, attributes = get_data('ClevelandData.csv')

normal_data = standardize_data(input_data)

# Split Data
training_data, test_data = data_split(normal_data, TEST_PERCENT)
# Define K-Means
kmeans = KMeans(n_clusters = int(max(training_data[LABEL])), random_state = 0)
# K-Means
attributes_without_label = copy.copy(attributes)
attributes_without_label.pop(attributes_without_label.index(LABEL))
kmeans.fit(np.array([training_data[key] for key in attributes_without_label]).T)
training_labels = kmeans.labels_

MAX_LABEL = int(max(training_data[LABEL]))

#print(training_data)
test = {'hek' : [1, 1, 1, 0],
        'num' : [0, 0, 1, 1]}
#MAX_LABEL = int(max(test[LABEL]))

#print(entropy('hek', test))

#print( ((.667) * np.log2(.667)) + ((.334) * np.log2(.334)))

for attribute in attributes_without_label:
    print(attribute)
    print(entropy(attribute, training_data))
    #entropy(attribute, training_data)
    print()