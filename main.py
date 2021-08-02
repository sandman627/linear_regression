import random
import math

sample_data = [[4, 8, 1, 5, 5, 4, 7, 570],
               [5, 4, 8, 7, 7, 2, 3, 563],
               [3, 8, 4, 7, 4, 5, 7, 521],
               [9, 1, 2, 9, 1, 5, 3, 610],
               [5, 2, 5, 9, 9, 1, 8, 593]]

data_size = 5
feature_num = 7
col_num = 8
list_parameters = []
bias = 0.001


for i in range(col_num):
    list_parameters.append(random.randint(1, 10))
print(list_parameters)

def hypothesis(list_data):
    prediction = list_parameters[0]
    for i in range(feature_num):
        prediction += list_parameters[i+1] * list_data[i]
    return prediction

def cost_function(): # squared error function
    sumoferror = 0
    for single_data in sample_data:
        sumoferror += math.pow(hypothesis(single_data) - single_data[len(single_data)-1], 2)
    return sumoferror / (2 * len(sample_data))

def pdf_cost_function(pdVariableNum):
    sumoferror = 0
    for single_data in sample_data:
        sumoferror = 2 * (hypothesis(single_data) - single_data[len(single_data)-1]) * single_data[pdVariableNum]
    return sumoferror

def gradient_descent():
    list_newParameters = []
    list_newParameters.append(list_parameters[0] - bias)
    for i in range(len(list_parameters)-1):
        list_newParameters.append(list_parameters[i + 1] - bias * pdf_cost_function(i) / data_size)
    return list_newParameters


def checking():
    list_prediction = []
    for single_data in sample_data:
        single_prediction = list_parameters[0]
        for i in range(feature_num):
            single_prediction += list_parameters[i+1] * single_data[i]
        list_prediction.append(single_prediction)
    print(list_prediction)


if __name__ == '__main__':
    print(sample_data)
    print(len(sample_data))
    print(cost_function())

    for i in range(100):
        list_parameters = gradient_descent()
        checking()

    print(list_parameters)



