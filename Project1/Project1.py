from math import sin , cos , pi , log
from random import random , randint
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import time

start = time.time()

# function to print a tree in vertical order (only used for debugging)
def printTree(node, level=0):
    if node != None:
        printTree(node.right, level + 1)
        print(' ' * 4 * level + '->', node.val)
        printTree(node.left, level + 1)

# plotting the results in a single figure
def plotting(x,y,approximated_y,Title):
    plt.plot(x,y,label = 'Main Function')
    plt.plot(x,approximated_y,label = 'Approximated Function')
    plt.title("Function = "+Title)
    plt.xlabel('x')
    plt.ylabel('f (x)')
    plt.legend()
    plt.show()


# store a binary tree node
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
 
 
# function to check if a given node is a leaf node
def is_leaf(node):
    return node.left is None and node.right is None
 
def is_single_operator(node):
    return  (node.left is None and node.right != None) or (node.right is None and node.left != None)

# function to evaluate a mathematical expression
def evaluator(expression,x_list):
    out_list = [None for _ in range(len(x_list))]
    for i in range(len(x_list)):
        x = x_list[i]
        try :
            out_list[i] = eval(expression)
            if(np.iscomplex(out_list[i])):
                out_list[i] = 1e9
        except :
            try :
                x = x_list[i] + 0.1
                out_list[i] = eval(expression)
                if(np.iscomplex(out_list[i])):
                    out_list[i] = 1e9
            except:
                out_list[i] = 1e9
        out_list[i] = out_list[i].real
    return out_list
 
 
# recursive function to construct an expression tree
def expression_constructor(root):
 
    # base case: invalid input
    if root is None:
        return 0
 
    # the leaves of a binary expression tree are operands
    if is_leaf(root):
        return root.val

    if is_single_operator(root):
        if root.left != None :
            x = expression_constructor(root.left)
            exp = "(" + str(root.val) + "(" +  str(x) + ")" + ")"
            return exp
        elif root.right != None :
            x = expression_constructor(root.right)
            exp = "(" + str(root.val) + "(" +  str(x) + ")" + ")"
            return exp
 
    # recursively evaluate the left and right subtree
    x = expression_constructor(root.left)
    y = expression_constructor(root.right)
 
    # make the expression in a string form
    exp = "(" + str(x) + str(root.val) + str(y) + ")"
    return exp
 


def random_generation_initialization(root,layer,Max_layers):
    flag = 0
    operator_list = ['+','-','*','/','**','sin','cos','abs']
    node_prob = random()
    if layer == Max_layers :
        if node_prob <= 0.5 :
            root = Node('x')
            flag = 2
        else:
            root = Node(str(randint(-5,5)))
            flag = 2
    else :
        N = len(operator_list) + 2
        if node_prob <= 1/N :
            root = Node('x')
            flag = 2
        elif node_prob > 1/N and node_prob <= 2/N :
            root = Node(str(randint(-5,5)))
            flag = 2
        elif node_prob > 2/N and node_prob <= 3/N :
            root = Node(operator_list[0])
        elif node_prob > 3/N and node_prob <= 4/N :
            root = Node(operator_list[1])
        elif node_prob > 4/N and node_prob <= 5/N :
            root = Node(operator_list[2])
        elif node_prob > 5/N and node_prob <= 6/N :
            root = Node(operator_list[3])
        elif node_prob > 6/N and node_prob <= 7/N :
            root = Node(operator_list[4])
        elif node_prob > 7/N and node_prob <= 8/N :
            root = Node(operator_list[5])
            flag = 1
        elif node_prob > 8/N and node_prob <= 9/N :
            root = Node(operator_list[6])
            flag = 1
        elif node_prob > 9/N and node_prob <= 10/N :
            root = Node(operator_list[7])
            flag = 1

    if flag == 0 and layer < Max_layers :
        layer = layer + 1
        root.left = random_generation_initialization(root.left, layer, Max_layers)
        root.right = random_generation_initialization(root.right, layer, Max_layers)
    elif flag == 1 and layer < Max_layers:
        layer = layer + 1
        p = random()
        if p <= 0.5 :
            root.left = random_generation_initialization(root.left, layer, Max_layers)
        else:
            root.right = random_generation_initialization(root.right, layer, Max_layers)
    return root

def fitness(population,y_list,x_list):
    efficiancy_flag = 0
    best_output = []
    best_function = ''
    best_output_fitness = ''
    scores = []
    counter = 0
    Min_MSE = 1e200
    for individual in population :
        MSE = 0
        out_list = evaluator(expression_constructor(individual),x_list)
        for i in range(len(out_list)):
            try :
                MSE = MSE + (y_list[i] - out_list[i])**2
            except OverflowError:
                MSE = 1e9
        if (MSE < Min_MSE) :
            Min_MSE = MSE
            best_output = out_list
            best_function = expression_constructor(individual)
            try:
                best_output_fitness = str(1/MSE)
            except :
                best_output_fitness = "Inf"

        scores.append(MSE)
        counter = counter + 1

    if (Min_MSE < 0.001):
        efficiancy_flag = 1

    return scores,efficiancy_flag,best_function,best_output,best_output_fitness


def selection (population,scores,method):
    if method == 'method1' :
        Scores_Sum = sum(scores)
        probabilities = [ (1-(score/Scores_Sum))/(len(scores)-1) for score in scores]
        population_number = 20
        selected_population = []
        for i in range(population_number):
            selection_prob = random()
            bios =  0
            for j in range(len(probabilities)):
                if  selection_prob > bios and selection_prob <= probabilities[j] + bios :
                    selected_population.append(population[j])
                bios = bios + probabilities[j]
    return selected_population


def crossover(selected_population):
    crossedover_population = []
    for k in range(len(selected_population)) :
        p1 = selected_population [k]
        for l in range(k+1,len(selected_population)) :
            p2 = selected_population [l]
            num = 20
            node1 = deepcopy(p1)
            tree1 = node1
            parent1 = node1
            child1 = ''
            child2 = ''
            for i in range(num):
                child_prob = random()
                if i!=0 and child_prob <= 1/3 :
                    break
                elif child_prob > 1/3  and child_prob <= 2/3 and node1.left != None:
                    parent1 = node1
                    child1 = 'left'
                    node1 = node1.left
                elif child_prob > 2/3 and node1.right != None:
                    parent1 = node1
                    child1 = 'right'
                    node1 = node1.right
                elif child_prob > 1/3  and child_prob <= 2/3 and node1.left == None:
                    break
                elif child_prob > 2/3 and node1.right == None:
                    break


            node2 = deepcopy(p2)
            tree2 = node2
            parent2 = node2
            for i in range(num):
                child_prob = random()
                if i!=0 and child_prob <= 1/3 :
                    break
                elif child_prob > 1/3  and child_prob <= 2/3 and node2.left != None:
                    parent2 = node2
                    child2 = 'left'
                    node2 = node2.left
                elif child_prob > 2/3 and node2.right != None:
                    parent2 = node2
                    child2 = 'right'
                    node2 = node2.right
                elif child_prob > 1/3  and child_prob <= 2/3 and node2.left == None:
                    break
                elif child_prob > 2/3 and node2.right == None:
                    break

            if (child1 == 'left' and child2 == 'left'):
                (parent1.left,parent2.left) = (parent2.left,parent1.left)
            elif (child1 == 'left' and child2 == 'right'):
                (parent1.left,parent2.right) = (parent2.right,parent1.left)
            elif (child1 == 'right' and child2 == 'right'):
                (parent1.right,parent2.right) = (parent2.right,parent1.right)
            elif (child1 == 'right' and child2 == 'left'):
                (parent1.right,parent2.left) = (parent2.left,parent1.right)

            crossedover_population.append(tree1)
            crossedover_population.append(tree2)

    return crossedover_population

def mutation(crossedover_population,mutation_propability):
    single_operators_list = ['sin','cos','abs']
    double_operators_list = ['+','-','*','/','**']
    next_generation = []

    for k in range(len(crossedover_population)):
        if (100*random()>mutation_propability):
            next_generation.append(crossedover_population[k])
            continue
        else:
            p1 = crossedover_population[k]
            tree = p1
            n = 20
            for i in range(n):
                    child_prob = random()
                    if child_prob <= 1/3:
                        break
                    elif child_prob > 1/3 and child_prob <= 2/3 and p1.left != None:
                        p1 = p1.left
                    elif child_prob > 2/3 and p1.right != None:
                        p1 = p1.right
                    elif child_prob > 1/3 and child_prob <= 2/3 and p1.left == None:
                        break
                    elif child_prob > 2/3 and p1.right == None:
                        break

            if is_single_operator(p1):
                r = int((len(single_operators_list)-1)*random())
                Mutated_p1 = single_operators_list[r]
                while(Mutated_p1 == p1.val):
                    r = int((len(single_operators_list)-1)*random())
                    Mutated_p1 = single_operators_list[r]
                p1.val = Mutated_p1

            elif is_leaf(p1):
                Mutated_p1 = 100-200*random()
                while(Mutated_p1 == p1.val):
                    Mutated_p1 = 100-200*random()
                p1.val = Mutated_p1

            else :
                r = int((len(double_operators_list)-1)*random())
                Mutated_p1 = double_operators_list[r]
                while(Mutated_p1 == p1.val):
                    r = int((len(double_operators_list)-1)*random())
                    Mutated_p1 = double_operators_list[r]
                p1.val = Mutated_p1

        next_generation.append(tree)
    return next_generation


def GP_function_approximator(initial_population_number,Max_generation,mutation_probability,x_list,y_list):
    root = Node('+')
    Max_layer = 2
    population = [random_generation_initialization(root,0,randint(1,3)) for _ in range(initial_population_number)]
    current_generation = 0
    while(current_generation < Max_generation):
        print("process "+ str(100*current_generation/Max_generation) +str("%")+ " completed")
        scores,efficiancy_flag,best_function,best_output,best_output_fitness = fitness(population,y_list,x_list)
        if efficiancy_flag == 1 :
            break
        else:
            selected_population = selection (population,scores,'method1')
            crossedover_population = crossover(selected_population)
            next_generation = mutation(crossedover_population,mutation_probability)
            population = next_generation
        current_generation = current_generation + 1
        mutation_probability = max(100*current_generation/Max_generation , mutation_probability)
    return current_generation, best_function, best_output, best_output_fitness




if __name__ == '__main__':

    ## Initializations
    initial_population_number = 100
    Max_generation =  20
    mutation_probability = 10

    ## Inputs
    Func = 'x'
    x_list = []
    y_list = []
    for i in range(1,100):
        x_list.append(i)
        y_list.append(i)


    # Main code
    current_generation, best_function, best_output, best_output_fitness = GP_function_approximator(initial_population_number,Max_generation,mutation_probability,x_list,y_list)

    #Results
    print("approximated function = ",best_function)
    print("fitness = ",best_output_fitness)
    print(str(current_generation)+" generations were passed")

    execution_duration = time.time()-start
    print("execution_duration = ",str(execution_duration),"s")

    plotting(x_list,y_list,best_output,Func)


