__author__ = 'jayanthvenkataraman'

import operator
from os import listdir

from numpy import tile,array,zeros,shape
import matplotlib


def classify(input_data_instance, data_set, labels, k):
    data_set_size = data_set.shape[
        0]  # shape on a matrix gives a tuple (n,m) where n is the number of rows of the matrix and m the column size

    input_data_instance_matrix = tile(input_data_instance, (data_set_size, 1))
    diff_matrix = input_data_instance_matrix - data_set

    square_diff_matrix = diff_matrix ** 2
    square_distances = square_diff_matrix.sum(axis=1)
    distances = square_distances ** 0.5

    sorted_distance_indices = distances.argsort() #returns indices of the array to be sorted

    class_count = {}
    for i in range(k):
        label_of_the_data_instance = labels[sorted_distance_indices[i]]
        class_count[label_of_the_data_instance] = class_count.get(label_of_the_data_instance, 0) + 1

    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def file_to_matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}

    with open(filename) as f:
        array_of_lines = f.readlines()
    number_of_lines = len(array_of_lines)         #get the number of lines in the file
    return_matrix = zeros((number_of_lines,3))    #prepare matrix to return
    class_label_vector = []                       #prepare labels return

    index = 0
    for line in array_of_lines:
        list_from_line = line.strip().split('\t')
        return_matrix[index,:] = list_from_line[0:3]

        label_for_this_data_instance = int(list_from_line[-1]) if list_from_line[-1].isdigit() else love_dictionary.get(list_from_line[-1])
        class_label_vector.append(label_for_this_data_instance)
        index += 1

    return return_matrix,class_label_vector

def auto_normalize_data(data_set):
    min_values = data_set.min(0) #gets the minimum values of each columns
    max_values = data_set.max(0) #gets the maximum values of each columns
    ranges = max_values - min_values

    normalized_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    normalized_data_set = data_set - tile(min_values, (m,1))
    normalized_data_set = normalized_data_set/tile(ranges, (m,1))   #element wise divide
    return normalized_data_set, ranges, min_values

def dating_class_test():
    hold_out_ratio = 0.50      #hold out 10% ideally
    dating_matrix,dating_labels = file_to_matrix('datingTestSet2.txt')       #load data setfrom file
    normalized_data_set, ranges, min_values = auto_normalize_data(dating_matrix)
    m = normalized_data_set.shape[0]
    num_test_vector = int(m*hold_out_ratio)
    errorCount = 0.0
    for i in range(num_test_vector):
        classifierResult = classify(normalized_data_set[i,:],normalized_data_set[num_test_vector:m,:],dating_labels[num_test_vector:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, dating_labels[i])
        if (classifierResult != dating_labels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(num_test_vector))
    print errorCount

def classify_person():
    results_list = ['not at all', 'in small doses', 'in large doses']
    percent_video_games = float(raw_input("percentage of time spent playing video games?"))
    frequent_flier_miles = float(raw_input("frequent flier miles earned per year?"))
    ice_cream = float(raw_input("liters of ice cream consumed per year?"))
    dating_matrix, dating_labels = file_to_matrix('datingTestSet2.txt')
    normalized_data_set, ranges, min_values = auto_normalize_data(dating_matrix)
    input_arr = array([frequent_flier_miles, percent_video_games, ice_cream, ])
    classifier_result = classify((input_arr - min_values)/ranges, normalized_data_set, dating_labels, 3)
    print "You will probably like this person: %s" % results_list[classifier_result - 1]

def img_to_vector(filename):
    return_vector = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vector[0,32*i+j] = int(line_str[j])
    return return_vector

def handwriting_class_test():
    hw_labels = []
    training_file_list = listdir('trainingDigits')  #load the training set
    m = len(training_file_list)
    training_matrix = zeros((m,1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]     #take off .txt
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_matrix[i,:] = img_to_vector('trainingDigits/%s' % file_name_str)
    test_file_list = listdir('testDigits')        #iterate through the test set
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]     #take off .txt
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img_to_vector('testDigits/%s' % file_name_str)
        classifier_result = classify(vector_under_test, training_matrix, hw_labels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifier_result, class_num_str)
        if (classifier_result != class_num_str): error_count += 1.0
    print "\nthe total number of errors is: %d" % error_count
    print "\nthe total error rate is: %f" % (error_count/float(m_test))
#group, labels = create_data_set()
#print classify([0, 0], group, labels, 3)

#dating_matrix,dating_labels = file_to_matrix("datingTestSet.txt")
#print dating_matrix[:,2]

#dating_class_test()
#classify_person()
#print img_to_vector("digits/testDigits/0_13.txt")

#handwriting_class_test()