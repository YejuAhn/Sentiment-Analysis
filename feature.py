#Goal : implement Natural Language Processing (NLP) system
#feature.py : produce a sparse representation of data using label-index-value format
import sys
import csv
from collections import OrderedDict


#store dictionary
def get_dict(dict_input):
    f = open(dict_input, 'r')
    input_f = csv.reader(f, delimiter=' ')
    dictionary = OrderedDict()
    for ith_row in input_f:
        dictionary[ith_row[0]] = ith_row[1]
    return dictionary

#Model1: store data and transform into sparse representation
def model1_to_sparse(input_file, output_file ,dictionary):
    f = open(input_file, 'r')
    out = open(output_file, 'w')
    input_f = csv.reader(f, delimiter= '\t')
    for ith_row in input_f:
        label = ith_row[0]
        words = ith_row[1]
        spilt_words = words.split()
        txt = str(label) + "\t"
        for key in dictionary:
            if key in spilt_words:
                index = dictionary.get(key)
                txt += str(index) + ":" + str(1) + "\t"
        txt = txt[:-1]
        txt += "\n"
        out.write(txt)
    out.close()


#Model2: store data and transform into sparse representation
def model2_to_sparse(input_file, output_file ,dictionary, thresh):
    f = open(input_file, 'r')
    out = open(output_file, 'w')
    input_f = csv.reader(f, delimiter= '\t')
    for ith_row in input_f:
        label = ith_row[0]
        words = ith_row[1]
        txt = str(label) + "\t"
        spilt_words = words.split()
        for key in dictionary:
            if key in spilt_words and spilt_words.count(key) < thresh:
                index = dictionary.get(key)
                txt += str(index) + ":" + str(1) + "\t"
        txt = txt[:-1]
        txt += "\n"
        out.write(txt)
    out.close()

if __name__ == '__main__':
    #path to the input .tsv file
    train_input = sys.argv[1]
    valid_input = sys.argv[2]
    test_input = sys.argv[3]
    #dictionary
    dict_input = sys.argv[4]
    #path to the output .tsv file
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    #model
    feature_flag = int(sys.argv[8])
    #get dictionary
    dictionary = get_dict(dict_input)
    if feature_flag == 1:
        #model 1
        model1_to_sparse(train_input, formatted_train_out, dictionary)
        model1_to_sparse(valid_input, formatted_validation_out, dictionary)
        model1_to_sparse(test_input, formatted_test_out, dictionary)
    else:
        #model 2
        thresh = 4
        model2_to_sparse(train_input, formatted_train_out, dictionary, thresh)
        model2_to_sparse(valid_input, formatted_validation_out, dictionary, thresh)
        model2_to_sparse(test_input, formatted_test_out, dictionary, thresh)











