from copy import deepcopy
from math import ceil, sqrt
import numpy as np

def find_duplicates(array):
    unique_rows, unique_indices, counts = np.unique(array, axis=0, return_index=True, return_counts=True)
    duplicate_indices = np.where(counts > 1)[0]
    duplicate_indices_list = []
    unique_elements_list = []
    for index in duplicate_indices:
        indices = np.where(np.all(array == unique_rows[index], axis=1))[0]
        duplicate_indices_list.append(indices)
        unique_elements_list.append(array[indices[0]])
    for i, index in enumerate(unique_indices):
        if i not in duplicate_indices:
            duplicate_indices_list.append([index])
            unique_elements_list.append(array[index])
    return np.array(duplicate_indices_list), np.array(unique_elements_list)

def domCheck(pop1,pop2):
    # check if obj1 dominates obj2. returns 1 or 0.
    isDominated = 1
    if isinstance(pop1,float):
        if pop1<=pop2:
                isDominated = 0
        return isDominated
    else:
        for obj_index in range(len(pop1)):
            if pop1[obj_index]<pop2[obj_index]:
                isDominated = 0
        if isDominated == 1:
            isEqual = 1
            for obj_index in range(len(pop1)):
                if pop1[obj_index]!=pop2[obj_index]:
                    isEqual = 0
                    break
            if isEqual == 1:
                isDominated = 0
        return isDominated
    

def myFirstPareto(array, comb):
    # accepts a numpy array, rows are population, columns are objectives. Change the columns if you want to change objectives.
    # makes a lookup table, if an element is dominated by the 0th array pop, the lookup table index becomes 0 and it's not checked anymore.
    red_array = array[:,comb]
    num_loc = len(red_array)
    lookup = np.ones([num_loc])
    res_array=[]

    # utopia check
    for pop in range(num_loc):
        if np.all(red_array[pop] == 1):
            res_array.append([pop,comb,red_array[pop]])
    if len(res_array) != 0:
        return res_array

    for pop1 in range(num_loc):
        if lookup[pop1] == 1: # if the first element is still in the front
            for pop2 in range(num_loc):
                if (pop1 != pop2) and (lookup[pop2] == 1):
                    domRes = domCheck(red_array[pop1],red_array[pop2])
                    if domRes == 1:
                        lookup[pop2] = 0
        
        """if (pop1+1) % (num_loc) == 0:
            print("Completed the combination",comb)"""
    for index in range(num_loc):
        if lookup[index] == 1:
            res_array.append([index,comb,red_array[index]])
    return res_array

def merge_pareto(array,comb):
    merged_indices = []
    for i in range(ceil(sqrt(len(array)))):
        below_index = i*ceil(sqrt(len(array)))
        upper_index = below_index+ceil(sqrt(len(array)))+1
        if upper_index > len(array):
            upper_index = len(array)
        results = myFirstPareto(array[below_index:upper_index,:],comb)
        for j in range(len(results)):
            merged_indices.append(results[j][0]) # add the indices to the complete list
    print(comb)
    final_result = myFirstPareto(array[merged_indices,comb])
    return final_result

def try_add_sorting(array,individual):
    #try adding individual to an existing sorting list [[index,value],[index,value]]
    for i in range (len(array)):
        if individual[1] > array[i][1]:
            for j in range(len(array)-i):
                if j!=0:
                    array[len(array)-j]=deepcopy(array[len(array)-1-j])
            array[i]=deepcopy(individual)
            break

def write_array(array, file_name):
    with open(file_name, 'w') as f:
        for row in array:
            f.write(' '.join(map(str, row)) + '\n')

def read_array(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        row_data = line.strip().split()
        converted_row_data = []
        for item in row_data:
            try:
                converted_row_data.append(int(item))
            except ValueError:
                try:
                    converted_row_data.append(float(item))
                except ValueError:
                    converted_row_data.append(item)
        data.append(converted_row_data)
    data_array = np.array(data, dtype=object)
    return data_array