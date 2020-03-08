
import csv

# attributes = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
# hi
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
        for entry in reader:
            DIM = len(entry)
            temp = []
            for i in range(len(entry)):
                if(entry[i] == '?'):
                    data[attributes[i]].append(entry[i])
                else:
                    try:
                        data[attributes[i]].append(float(entry[i]))
                    except:
                        data[entry[i]] = []
                        attributes.append(entry[i])
    replace_w_median(data, attributes)
    return data

def replace_w_median(data, attributes):
    for title in attributes:
        for i in range(len(data[title])):
            middle = int(len(data[title])/2)
            if data[title][i] == '?':
                while (data[title][middle] == '?'):
                    middle += 1
                data[title][i] = data[title][middle]


get_data('ClevelandData.csv')
