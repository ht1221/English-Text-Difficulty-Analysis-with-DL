# coding:utf-8
__author__ = 'lenovo-gjfht-real'
import numpy as np
import os
m_path = "C:\\Users\\lenovo-gjfht-real\\Desktop\\ETDAD\\word_seq"
outfile_name1 = "sentences.txt"
outfile_name2 = "difficulty.txt"
error_distribution_path_for_svm = "error_distribution.txt"      
error_distribution_path_for_lstm = "error_distribution_for_lstm.txt"    

MAX_WRONG_TIME_ALLOWED_FOR_SVM = 7

MAX_WRONG_TIME_ALLOWED_FOR_KNN = 5
name2id = {'Cet4_word_seq': 2, 'Cet6_word_seq': 3, 'NMET_word_seq': 1, 'ZK_word_seq': 0}
name2id_ = {'Cet4': 2, 'Cet6': 3, 'NMET': 1, 'ZK': 0}
id2name = {0: "ZK", 1: "NMET", 2: "Cet4", 3: "Cet6"}


def load_data_xy(path_s, path_d):
   
    data_x, data_y = [], []
    f_s = open(name=path_s, mode='rb')
    for sent in f_s:
        data_x.append(sent.strip().split(' '))
    f_s.close()
    f_d = open(name=path_d, mode='rb')
    for label in f_d:
        data_y.append(int(label))
    f_d.close()

    return data_x, data_y


def load_data_xyz(path_s, path_d):
   
    x_, y_ = load_data_xy(path_s, path_d)
    z_ = []
    for i in range(len(y_)):
        z_.append((y_[i], i))

    return x_, z_


def sum_wrong_index(path_w, max_index, max_allowed):
   
    w_index_list = []
    count = np.zeros(max_index)
    f = open(path_w, 'rb')
    for line in f:
        index = int(line.strip())
        w_index_list.append(index)
    f.close()
    for item in w_index_list:
        count[item] += 1
    dic = {}
    sample_passed = []
    for i in range(max_index):
        if count[i] in dic.keys():
            dic[count[i]] += 1
        else:
            dic[count[i]] = 1
            
        if count[i] <= max_allowed:      
            sample_passed.append(i)
            
    print dic
    return dic, sample_passed


def less_than(limit, dic):
    
    count = 0
    for key in range(limit):
        count += dic[key]
    return count

wrong_dic, sample_allowed_for_svm = sum_wrong_index('wrong.txt', 26787, MAX_WRONG_TIME_ALLOWED_FOR_SVM)


def get_hand_operated_features(x_, index2word_, dict_most2000):
    '''
    :param x_ word index seq of every sentence
    :param index2word_ a dict, index -> word
    :return 3 hand-operated features of every sentence，including average character number, sentence length, the ratio of commom words
    '''
    features = np.zeros((len(x_), 3), dtype="float32")
    for i in range(len(x_)):
        line = x_[i]    
        character_counts = 0
        commom_word_counts = 0
        sentence_length = 0
        for j in range(len(line)):
            if line[j] == 0:
                break
            sentence_length += 1
            word = index2word_[line[j]]    
            character_counts += len(word)
            if word in dict_most2000.keys():
                commom_word_counts += 1
        features[i][0] = character_counts / sentence_length
        features[i][1] = sentence_length
        features[i][2] = commom_word_counts / sentence_length
    return features


def record_error_distribution(labels, errors, path_ed):
    
    error_distribution = {}
    if len(labels) != len(errors):
        print "input error"
        return
    length = len(labels)
    for i in range(length):
        name = id2name[labels[i]]
        if name in error_distribution:
            error_distribution[name] += 1
        else:
            error_distribution[name] = 0
            error_distribution[name + "_nError"] = 0
        if not errors[i]:
            error_distribution[name + "_nError"] += 1
    f_out_ed = open(path_ed, 'a')
    for key in error_distribution.keys():
        content = key + " " + str(error_distribution[key]) + "\n"   
        f_out_ed.write(content)


def read_and_compute_error_distribution(path_ed):
    
    count1 = [0, 0, 0, 0]
    count2 = [0, 0, 0, 0]   
    lines = open(path_ed, "rb").readlines()
    for line in lines:
        key, count = line.strip().split(' ')[0], int(line.strip().split(' ')[1])    # 划分每一行为一个关键字key和一个计数值
        if key[-2:] == "or":
            level = name2id_[key[0:-7]]
            count2[level] += count
        else:
            level = name2id_[key]
            count1[level] += count
    for i in range(4):
        print "%s一共在测试集中错误 %d(%d) 次, 错误率是 %f" % (id2name[i], count2[i], count1[i], float(count2[i])/float(count1[i]))



