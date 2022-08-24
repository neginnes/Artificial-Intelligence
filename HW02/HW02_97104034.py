from math import log2
from random import randint
from copy import deepcopy
import numpy as np
import csv
import graphviz
import random




class Node:
    def __init__(self, parent = None , classification_name = "", decision = None, child=[], node_string_number = "" , Entropy = None , Gain = None ):
        self.classification_name = classification_name
        self.Entropy = Entropy
        self.Gain = Gain
        self.decision = decision
        self.child = child
        self.parent = parent
        self.node_string_number = node_string_number
    def Node_Number(self,string_number):
        self.node_string_number = string_number


def find(List,X):
    return [i for i,x in enumerate(List) if x==X]

def column(matrix, col): 
    return [row[col] for row in matrix]

def del_column(matrix, col): 
    rows = []
    for row in matrix :
        del row[col]
        rows.append(row)
    return rows

def Binary_Entropy(q):
    if q == 0 or q == 1 :
        H = 0
    else:
        H = -(q*log2(q)+(1-q)*log2(1-q))
    return H


def gain(data,i_th_attribute_index):
    goals = column(data,len(data[0])-1)
    p = len(find(goals,1))
    n = len(find(goals,0))
    B = Binary_Entropy(p/(p+n))
    i_th_data = column(data,i_th_attribute_index)
    M = set(i_th_data)
    Number_of_repetitions = []
    Bk = []
    for i in M:
        Number_of_repetitions.append(len(find(i_th_data,i)))
        nk = 0
        pk = 0
        for j in range(len(i_th_data)):
            if i_th_data[j]==i and goals[j] == 1 :
                pk = pk + 1
            elif i_th_data[j]==i and goals[j] == 0:
                nk = nk + 1
        Bk.append(Binary_Entropy(pk/(pk+nk)))
    Remainder = 0
    for i in range(len(Number_of_repetitions)):
        Remainder = Remainder + (Number_of_repetitions[i]/len(i_th_data))*Bk[i]

    return B - Remainder



def most_important_attribute(data,attributes):
    Gain = -1
    for i in range(len(attributes)-1):
        if gain(data,i) > Gain :
            Gain = gain(data,i)
            i_th = i
    return i_th

def plurality_value(data):
    goals = column(data,len(data[0])-1)
    p = len(find(goals,1))
    n = len(find(goals,0))
    if(p>n):
        PV = "yes"
    elif(p<n):
        PV = "no"
    else: 
        decisions = ["yes","no"]
        PV = decisions[randint(0,1)]
    return PV

def data_divider(data, data_strings, i_th_attribute_index, M):
    divided_data = []
    divided_data_strings = []
    for k in M:
        k_th_data_cluster = []
        k_th_data_strings_cluster = []
        for i in range(len(data)):
            if data[i][i_th_attribute_index] == k :
                k_th_data_cluster.append(data[i])
                k_th_data_strings_cluster.append(data_strings[i])

        divided_data.append(k_th_data_cluster)
        divided_data_strings.append(k_th_data_strings_cluster)
    return divided_data_strings, divided_data



def decision_tree_creator(parent_node, attributes, data, data_strings, classification_name, decision, Plurality_Value, Gain):
    goals = column(data,len(data[0])-1)
    p = len(find(goals,1))
    n = len(find(goals,0))
    B = Binary_Entropy(p/(p+n))
    if len(data) == 0:
        new_node = Node(parent_node, classification_name, Plurality_Value, [], "", B, Gain)
        parent_node.child.append(new_node)
        return
    elif len(attributes) == 1 :
        new_node = Node(parent_node, classification_name, plurality_value(data), [], "", B, Gain)
        parent_node.child.append(new_node)
        return
    elif len(find(goals,1)) == len(goals):
        new_node = Node(parent_node, classification_name, "yes", [], "", B, Gain)
        parent_node.child.append(new_node)
        return
    elif len(find(goals,0)) == len(goals):
        new_node = Node(parent_node, classification_name, "no", [], "", B, Gain)
        parent_node.child.append(new_node)
        return

    else:
        Plurality_Value = plurality_value(data)
        i_th_attribute_index = most_important_attribute(data,attributes)
        decision = attributes[i_th_attribute_index]
        Gain = gain(data,i_th_attribute_index)
        new_node = Node(parent_node, classification_name, decision, [], "", B,Gain)
        parent_node.child.append(new_node)
        i_th_data = column(data,i_th_attribute_index)
        M = set(i_th_data)
        divided_data_strings, divided_data = data_divider(data, data_strings, i_th_attribute_index, M)
        del attributes[i_th_attribute_index]
        idx = 0
        for k in M:
            new_data = divided_data[idx]
            new_data_strings = divided_data_strings[idx]
            k_th_cluster_indices = find(i_th_data,k)
            classification_name = data_strings[k_th_cluster_indices[0]][i_th_attribute_index]
            new_data = del_column (new_data,i_th_attribute_index)
            new_data_strings = del_column (new_data_strings,i_th_attribute_index)
            decision_tree_creator(new_node, deepcopy(attributes), deepcopy(new_data), deepcopy(new_data_strings), classification_name, decision, Plurality_Value,Gain)
            idx = idx + 1

def visualization(root):
    tree_leaves = [root]

    dot = graphviz.Digraph('Decision Tree', comment='Decision Tree')

    number = 0
    string_number = str(number)
    root.Node_Number(string_number)

    while tree_leaves != []:
        for leaf in tree_leaves[0].child:
            if tree_leaves[0].node_string_number == "0":
                number = number + 1
                string_number = str(number)
                dot.node(leaf.node_string_number, leaf.decision + " (Entropy = " + str(leaf.Entropy) + ")")  
            else:
                number = number + 1
                string_number = str(number)
                leaf.Node_Number(string_number)
                dot.node(leaf.node_string_number, leaf.decision + " (Entropy = " + str(leaf.Entropy) + ")")  
                start = tree_leaves[0].node_string_number
                end = leaf.node_string_number
                # print(start, end)
                dot.edge((start), (end),"     [" + str(leaf.classification_name) + ") (Gain = " + str(leaf.Gain) + ")            ")  
            tree_leaves.append(leaf)
        tree_leaves.pop(0)

    dot.render(directory='doctest-output', view= True).replace('\\', '/')
    return

def input_reader (csv_input_file_name):
    file = open(csv_input_file_name)
    csvreader = csv.reader(file)
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    attributes = header
    data = rows
    return attributes, data

def continuous_to_discrete(continuous_data_strings,number_of_groups):
    continuous_data = [[None for _ in range(len(continuous_data_strings[0]))] for _ in range(len(continuous_data_strings))]
    for i in range(len(continuous_data_strings)):
        for j in range(len(continuous_data_strings[0])):
            continuous_data[i][j] = float(continuous_data_strings[i][j] )

    steps = []
    ranges = []
    discrete_data_strings = [[None for _ in range(len(continuous_data[0]))] for _ in range(len(continuous_data))]
    for i in range(len(continuous_data[0])-1):
        i_th_column = column(continuous_data,i)
        Range = ( min(i_th_column) ,max(i_th_column) )
        ranges.append(Range)
        steps.append( 1+ ( ((Range[1]-Range[0]) ) / (number_of_groups[i]-2)) )

    for j in range(len(continuous_data[0])-1):
        Min = ranges[j][0]
        Max = ranges[j][1]
        step = steps[j]
        start = Min
        while(start < Max):
            for i in range(len(continuous_data)):
                if (float(continuous_data[i][j]) < start + step  and float(continuous_data[i][j]) >= start ) :
                    discrete_data_strings[i][j] =  str(start) + "," + str(start + step) 
            start = start + step

        for i in range(len(continuous_data)):
                if float(continuous_data[i][j]) >= start :
                    discrete_data_strings[i][j] = str(start)  + "," + "inf"
                elif float(continuous_data[i][j]) < Min :
                    discrete_data_strings[i][j] =   "-inf" + "," + str(Min)

    for i in range(len(continuous_data)):
        discrete_data_strings[i][len(continuous_data[0])-1] = continuous_data_strings[i][len(continuous_data[0])-1]
    
    return discrete_data_strings


def data_encoder(data_strings):
    encoded_data = [[None for _ in range(len(data_strings[0]))] for _ in range(len(data_strings))]
    for j in range(len(data_strings[0])):
        if j != len(data_strings[0])-1 :
            c = -1
            for i in range(len(data_strings)):
                if encoded_data[i][j] == None :
                    string = data_strings[i][j]
                    c = c + 1
                    encoded_data[i][j] = c
                for k in range (i+1,len(data_strings)):
                    if data_strings[k][j] == string :
                        encoded_data[k][j] = c
        else :
            for i in range(len(data_strings)):
                if data_strings[i][j] == "yes" or data_strings[i][j] == "1" :
                    encoded_data[i][j] = 1
                elif data_strings[i][j] == "no" or data_strings[i][j] == "0" :
                    encoded_data[i][j] = 0
    return encoded_data

def learn_data_picker(data,percentage):
    number_of_learn_data = int(percentage*(len(data))/100)
    random_participants = random.sample(range(0, len(data)), number_of_learn_data)
    learn_data = []
    test_data = []
    for i in range(len(data)):
        if i in random_participants :
            learn_data.append(data[i])
        else:
            test_data.append(data[i])
    return learn_data,test_data

def find_answer(attributes, single_test, node):
    if node.node_string_number == "0" :
        return find_answer(attributes, single_test, node.child[0])
    elif node.decision == "yes" :
        return "1"
    elif node.decision == "no" :
        return "0"
    elif len(node.child) == 0 :
        return str(randint(0,1))
    else :
        for i in range(len(attributes)):
            if attributes[i] == node.decision :
                attribute_index = i
        for k in range(len(node.child)):
            Range = [float (list(node.child[k].classification_name.split(","))[0]) , 
            float (list(node.child[k].classification_name.split(","))[1]) ]
            if "-inf" in node.child[k].classification_name:
                if (float(single_test[attribute_index]) < Range[1]):
                    return find_answer(attributes, single_test, node.child[k])
            elif "inf" in node.child[k].classification_name: 
                if (float(single_test[attribute_index]) >= Range[0]):
                    return find_answer(attributes, single_test, node.child[k])
            elif ((float(single_test[attribute_index]) >= Range[0]) and (float(single_test[attribute_index]) < Range[1])) :
                return find_answer(attributes, single_test, node.child[k])
                

def tester (attributes,test_data,root):
    test_results = []
    for single_test in test_data :
        string = find_answer(attributes, single_test, root)
        if string == None :
            string = str(randint(0,1))
        test_results.append(string)
    return test_results

def accuracy (test_results,test_goals):
    TP = 0
    TN = 0
    Total = len(test_results)
    for i in range(len(test_results)):
        if test_results[i] == test_goals[i]:
            TP = TP + 1
        elif test_results[i] == test_goals[i]:
            TN = TN + 1
    Accuracy = (TP + TN)/Total
    return Accuracy*100


if __name__ == '__main__':

    Input = "Q1"       # Q1 or Q2 ?
    if Input == "Q1" :
        attributes, data_strings = input_reader("Part1.csv")

        data = data_encoder(data_strings)
        root = Node()
        classification_name = ""
        decision = ""
        Plurality_Value = ""
        Gain = None
        decision_tree_creator(root, attributes, data, data_strings, classification_name, decision, Plurality_Value, Gain)
        visualization(root)

    elif Input == "Q2" :
        percentage = 50       # percentage of learn data
        Ranges =  4             # How many ranges?
        attributes, continuous_data_strings = input_reader("diabetes.csv")


        continuous_learn_data_strings, continuous_test_data_strings = learn_data_picker(continuous_data_strings,percentage)
        number_of_groups = [Ranges for _ in range(len(attributes)-1)]
        discrete_learn_data_strings  = continuous_to_discrete(continuous_learn_data_strings,number_of_groups)
        learn_data = data_encoder(discrete_learn_data_strings)
        root = Node()
        classification_name = ""
        decision = ""
        Plurality_Value = ""
        Gain = None
        decision_tree_creator(root, deepcopy(attributes), deepcopy(learn_data), deepcopy(discrete_learn_data_strings), classification_name, decision, Plurality_Value, Gain)
        visualization(root)
        test_results = tester (deepcopy(attributes),continuous_test_data_strings,root)
        test_goals = column(continuous_test_data_strings,len(attributes)-1)
        Accuracy = accuracy(test_results,test_goals)
        print("Accuracy is ", Accuracy , "%")






 




