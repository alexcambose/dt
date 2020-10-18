# ID3 algorithm
# Alexandru Cambose
import math


def list_without_element(array, element):
    """Returns a new list that doesn't contain the element"""
    copy = array.copy()
    copy.remove(element)
    return copy


def filter_dataset(dataset, attribute, value):
    """Returns a new dataset that contains the attributes
    that are connected with the value of the specified attribute
    """
    new_dataset = {}
    # Initialise each column with an empty array
    for attr in dataset:
        new_dataset[attr] = []

    for i in range(0, len(dataset[attribute])):
        # Keep only the rows that are corresponding with the 'value' arg
        if dataset[attribute][i] == value:
            for attr in dataset:
                new_dataset[attr].append(dataset[attr][i])
    return new_dataset


def calculate_entropy(data):
    """Calculates the entropy for a given array"""
    unique_values = list(set(data))
    data_length = len(data)
    entropy = 0
    for value in unique_values:
        count = data.count(value)
        # entropy with the sum formula, with reversed log because of the -
        entropy += (count / data_length) * math.log2(data_length / count)
    return entropy


def calculate_ig(data, x_label, y_label):
    """Calculates the information gain with: """
    x_array = data[x_label]
    y_array = data[y_label]
    y_values = list(set(y_array))
    ig_result = calculate_entropy(x_array)
    # for each value of the y attribute
    for value in y_values:
        # get the data on one side of the decision tree, the current node will have a y_values number of children
        x_partial_data = [x_array[i] for i in range(0, len(y_array)) if y_array[i] == value]
        ig_result -= len(x_partial_data) / len(x_array) * calculate_entropy(x_partial_data)
    return ig_result


def ID3(dataset, data_attrs, target_attr):
    tree = {}
    root_label = None
    # if the entropy of the attribute that we want to predict is 0, all the values are the same and the answer is clear
    if calculate_entropy(dataset[target_attr]) == 0:
        # return a value, 0 for example
        return dataset[target_attr][0]

    # calculate information gains for each attribute and select the biggest attribute found as the root node
    biggest_ig_number = 0
    for data_attr in data_attrs:
        # calculate information gain
        current_ig = calculate_ig(dataset, target_attr, data_attr)
        # compare to see if this is bigger
        if biggest_ig_number < current_ig:
            biggest_ig_number = current_ig
            # select the root node
            root_label = data_attr

    # prepare to add children nodes
    tree[root_label] = {}
    # loop through the values of a column
    for label_value in list(set(dataset[root_label])):
        # provide the further calls of the algorithm with the rows associated with the current branch
        filtered_dataset = filter_dataset(dataset, root_label, label_value)
        # recursively call algo, with the the current attribute removed from the dataset
        tree[root_label][label_value] = ID3(filtered_dataset, list_without_element(data_attrs, root_label), target_attr)
    return tree


def predict(tree, values):
    # found an answer
    if type(tree) is str:
        return tree
    for tree_key in tree.keys():
        for value in list(values.keys()) + list(values.values()):
            if tree_key == value:
                return predict(tree[value], values)


# ======

outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny'.split(',')
temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild'.split(',')
humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal'.split(',')
windy = 'weak,strong,strong,weak,weak,weak,strong,weak,strong,weak,strong,weak,weak,strong'.split(',')
play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes'.split(',')
dataset = {"outlook": outlook, "temp": temp, "humidity": humidity, "windy": windy, "play": play}

tree = ID3(dataset, ['outlook', 'temp', 'humidity', 'windy'], 'play')

prediction = predict(tree, {
    "outlook": "sunny",
    "temp": "cool",
    "humidity": "high",
    "wind": "strong",
})

print(tree)
# {'outlook': {'sunny': {'humidity': {'normal': 'yes', 'high': 'no'}}, 'overcast': 'yes', 'rainy': {'windy': {'weak': 'yes', 'strong': 'no'}}}}

print(prediction)
# no
