# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 19:17:26 2022

@author: ranran

"""

import os
import numpy as np
import scipy as sp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
 

Protein_Atom = ['C','N','O','S']
Ligand_Atom = ['C','N','O','S','P','F','Cl','Br','I']
aa_list = ['ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','HSE','HSD','SEC',
           'ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','PYL']


test_file = open('../data/2016/name/test_name.txt')
pre_test_data = test_file.readlines()
test_data = eval(pre_test_data[0])
test_file.close()

train_file = open('../data/2016/name/train_name.txt')
pre_train_data = train_file.readlines()
train_data = eval(pre_train_data[0])
train_file.close()

all_file = open('../data/2016/name/all_name.txt')
pre_all_data = all_file.readlines()
all_data = eval(pre_all_data[0])
all_file.close()


#############################################################################################  
# extract coordinate code starts

def get_index(a,b):
    t = len(b)
    if a=='Cl':
        return 6
    if a=='CL':
        return 6
    if a=='Br':
        return 7
    if a=='BR':
        return 7
    
    for i in range(t):
        if a[0]==b[i]:
            return i
    return -1

def distance_of_two_points(p0,p1):
    temp = pow(p0[0]-p1[0],2) + pow(p0[1]-p1[1],2) + pow(p0[2]-p1[2],2)
    r_ij = pow(temp,0.5)
    return r_ij

def pocket_coordinate(name,P_atom,L_atom,cutoff):
    ########################################################################################
    '''
    this function extract the atom coordinates for one type atom-pair of a protein-ligand
    complex calld name.
    output is a coordinate file, the coordinate file has four columns, 
    the former three columns are the coordinate, the last column are 1 and 2 for protein
    and ligand atoms respectively. 
    (1) P_atom and L_atom are elements of Protein_Atom list and Ligand_Atom list shown above
    (2) cutoff is a real number to confine the distance of proteins around ligands
    (3) before this function, you need to prepare the PDBbind data
    '''
    ########################################################################################
    t1 = '../data/2016/refined-set/' + name + '/' + name + '_pocket.pdb'
    f1 = open(t1,'r')
    p_list = []
    for line in f1.readlines():
        if (line[0:4]=='ATOM')&(line[17:20] in aa_list ):
            atom = line[13:15]
            atom = atom.strip()
            index1 = get_index(atom,Protein_Atom)
            index2 = get_index(P_atom,Protein_Atom)
            if  index1 == index2 :
                p_list.append([float(line[30:38]) , float(line[38:46]) , float(line[46:54]) , 1 ])
    f1.close()

    t2 = '../data/2016/refined-set/' + name + '/' + name + '_ligand.mol2'
    f2 = open(t2,'r')
    l_list = []
    contents = f2.readlines()
    t3 = len(contents)
    start = 0
    end = 0
    for i in range(t3):
        if contents[i][0:13]=='@<TRIPOS>ATOM':
            start = i + 1
            continue
        if contents[i][0:13]=='@<TRIPOS>BOND':
            end = i - 1
            break
    for j in range(start,end+1):
        if contents[j][8:17]=='thiophene':
            print('thiophene',j)
        atom = contents[j][47:49]
        atom = atom.strip()
        index1 = get_index(atom,Ligand_Atom)
        index2 = get_index(L_atom,Ligand_Atom)
        if  index1 == index2:
            l_list.append([float(contents[j][17:26]) , float(contents[j][26:36]) , float(contents[j][36:46]) , 2 ])
    f2.close()
    
    p_list2 = []
    for p in p_list:
        for l in l_list:
            dis = distance_of_two_points(p, l)
            if dis <= cutoff:
                p_list2.append(p)
                break
        
    all_atom = np.array(p_list2 + l_list)
    filename = '../data/2016/euclid_atom_combination/'  + name + '/' + 'Protein_' + P_atom + '_' + 'Ligand_' + L_atom + '_coordinate.csv'
    np.savetxt(filename , all_atom , delimiter = ',')

def write_pocket_coordinate_to_file(start,end,cutoff):
    # start and end are index of data you will deal with
    for i in range(start,end):
        name = all_data[i]
        os.mkdir('../data/2016/euclid_atom_combination/' +name)
        for P_atom in Protein_Atom:
            for L_atom in Ligand_Atom:
                pocket_coordinate(name,P_atom,L_atom,cutoff)
                
# extract coordinate code ends
#############################################################################################   





#########################################################################################################
# from now on, the codes are to get the eigenvalue information

def judge_complex(point):
    ########################################################################################
    '''
    this function statistics the numbers of protein and ligand respectively
    return is 0 or 1, 0 for protein-only or ligand-only cases,
    1 represents the case where both protein and ligand are present
    '''
    ########################################################################################
    P_num = 0
    L_num = 0
    if len(point.shape) >1:
        for k in point:
                if  k[3] == 1:
                    P_num = P_num + 1
                elif k[3] == 2:
                    L_num = L_num + 1
    
    if P_num == 0 or L_num == 0:
        return 0
    else:
        return 1
    
def Euclid_distance_matrix(point):
    ########################################################################################
    '''
    this function returns the Euclid distance matrix of a point cloud input
    input is point cloud obteined by pocket_coordinate function, 
    take two points of input for instance,
    their fourth column of the data are judged firstly, 
    if they come from the same type, let their distance be large enough, 
    and if they come from a different type, take the true value
    '''
    ########################################################################################
    t = len(point)
    matrix = np.zeros( (t,t) )
    for i in range(t):
        for j in range(i+1,t):
            if  point[i][3] == point[j][3]:
                matrix[i][j] = 1000
                matrix[i][j] = 1000
            else:
                matrix[i][j] = distance_of_two_points(point[i],point[j])
                matrix[j][i] = matrix[i][j]
    return matrix


def get_final_complex(e_matrix,max_filtration):
    t = len(e_matrix)
    pre_simplex = []
    for i in range(t):
        pre_simplex.append( [ [i],0 ] )
    for i in range(t):
        for j in range(i+1,t):
            if e_matrix[i][j]<=max_filtration:
                temp = [ [i,j], e_matrix[i][j] ]
                pre_simplex.append(temp)
    simplex = sorted(pre_simplex,key=lambda x:(x[1]))
    return simplex

def get_simplex(final_simplex,filtration):
    simplex_list = []
    for item in final_simplex:
        if item[1] <= filtration:
            simplex_list.append(item[0])
        else:
            break
    return simplex_list

def get_incidence_matrix(simplex_list):  
    t = 0
    vertex = []
    simplex1_list = [] 
    for simplex in simplex_list:
        if len(simplex) == 1:
            t = t + 1
            vertex.append(simplex[0])
        elif len(simplex) == 2:
            simplex1_list.append(simplex)
    relation_matrix = np.ones((t,t))*1000
    for i in range(t):
        relation_matrix[i,i] = 0
    for j in simplex1_list:
        relation_matrix[j[0],j[1]] = 1
        relation_matrix[j[1],j[0]] = 1
    return relation_matrix

def dijkstra(start, incidence_matrix,k):
    passed = [start]
    nopass = [x for x in range(len(incidence_matrix)) if x != start]
    dis = incidence_matrix[start]
    
    while len(nopass):
        idx = nopass[0]
        for i in nopass:
            if dis[i] < dis[idx]: 
                idx = i
        if dis[idx] > k:
            for i in nopass:
                dis[i] = 1000
            break
        
        nopass.remove(idx)
        passed.append(idx)
        
        for i in nopass:
            if dis[idx] + incidence_matrix[idx][i] < dis[i]: 
                 dis[i] = dis[idx] + incidence_matrix[idx][i]
                             
    return dis

def get_path_dis_matrix(graph,k):
    t = graph.shape[0]
    path_matrix = np.zeros((t,t))
    for i in range(t):
        dis = dijkstra(i, graph, k)
        for j in range(i+1,t):
            path_matrix[i][j] = dis[j]
            path_matrix[j][i] = path_matrix[i][j]
    return path_matrix


def get_laplacian_matrix(path_matrix,k):
    t = path_matrix.shape[0]
    laplacian_matrix = np.zeros((t,t))
    for i in range(t):
        temp = path_matrix[i]
        count = 0
        for j in range(t):
            if temp[j] == k:
                laplacian_matrix[i][j] = -1
                count = count +1
        laplacian_matrix[i][i] = count
    return laplacian_matrix


def get_eigenvalue(matrix):
    values = np.linalg.eigvalsh(matrix)
    none_0 = []
    t = 0
    for item in values:
        if item <= 0.000000001:
            t = t + 1
        else:
            none_0.append(round(item,5))
    return t,none_0


def get_feature(name,max_filtration,step,k):
    ########################################################################################
    '''
    this function extract the non-zero eigenvalues for a protein-ligand complex calld name
    output are three txt files, each file takes a list as a unit, 
    each list stores the non-zero eigenvalues under the specified filter values and hopping 
    '''
    ########################################################################################
    n = int(max_filtration/step) 
    vertexmatrix1 = []    
    vertexmatrix2 = []
    vertexmatrix3 = [] 
    
    for P_atom in Protein_Atom: 
        for L_atom in Ligand_Atom:
            
            filename = '../data/2016/euclid_atom_combination/'  + name + '/' + 'Protein_' + P_atom + '_' + 'Ligand_' + L_atom + '_coordinate.csv'
            point_cloud = np.loadtxt(filename,delimiter = ',')
            if judge_complex(point_cloud) == 0:
                for item in range(n):
                    vertexmatrix1.append([])
                    vertexmatrix2.append([])
                    vertexmatrix3.append([])
                continue
            dis_matrix = Euclid_distance_matrix(point_cloud)
            final_simplex = get_final_complex(dis_matrix, max_filtration)
        
            for i in range(n):
                simplex_list = get_simplex(final_simplex, (i+1)*step)
                incidence_matrix = get_incidence_matrix(simplex_list)
                vertex_path_matrix = get_path_dis_matrix(incidence_matrix, k)
                
                ##### hopping 1                
                vertex_lap_matrix1 = get_laplacian_matrix(vertex_path_matrix, 1)
                vertex_hopping_1_eigenvalue = get_eigenvalue(vertex_lap_matrix1)
                vertexmatrix1.append(vertex_hopping_1_eigenvalue[1])
                   
                ##### hopping 2                 
                vertex_lap_matrix2 = get_laplacian_matrix(vertex_path_matrix, 2)
                vertex_hopping_2_eigenvalue = get_eigenvalue(vertex_lap_matrix2)
                vertexmatrix2.append(vertex_hopping_2_eigenvalue[1])
                
                ##### hopping 3
                vertex_lap_matrix3 = get_laplacian_matrix(vertex_path_matrix, 3)
                vertex_hopping_3_eigenvalue = get_eigenvalue(vertex_lap_matrix3)
                vertexmatrix3.append(vertex_hopping_3_eigenvalue[1])
    
    feature_file1 = '../data/2016/euclid_eigenvalue/' + name + '_hop1.txt'
    f = open(feature_file1,'w')
    f.writelines(str(vertexmatrix1))
    f.close()
    feature_file2 = '../data/2016/euclid_eigenvalue/' + name + '_hop2.txt'
    f = open(feature_file2,'w')
    f.writelines(str(vertexmatrix2))
    f.close()
    feature_file3 = '../data/2016/euclid_eigenvalue/' + name + '_hop3.txt'
    f = open(feature_file3,'w')
    f.writelines(str(vertexmatrix3))
    f.close()
    

def eigenvalue_to_file(start,end,max_filtration,step,k):
    for i in range(start,end):
        name = all_data[i]
        get_feature(name, max_filtration, step, k)
        print(i,name,'finish')

# get eigenvalue code ends
##############################################################################################################



###########################################################################################################
# feature generation code starts

def get_median(ls):
    if len(ls)==0:
        return 0
    return np.median(ls)

def get_mean(ls):
    if len(ls)==0:
        return 0
    return np.mean(ls)

        
def get_eigenvalue_median_and_mean(name,hop):
    # extract attributes from nonzero eigenvalues
    
    matrix1 = np.zeros((1,36*100))
    matrix2 = np.zeros((1,36*100))
    nonzero_file = '../data/2016/euclid_eigenvalue/' + name + '_hop' + str(hop) + '.txt'
    f = open(nonzero_file)
    pre_nonzero_eigenvalue = f.readlines()
    nonzero = eval(pre_nonzero_eigenvalue[0])
    f.close()
    count1 = 0
    count2 = 0
    for i in range(36*100):
        nonzero_list = nonzero[i]
        matrix1[0][count1] = get_median(nonzero_list)
        count1 = count1 + 1
        matrix2[0][count2] = get_mean(nonzero_list)
        count2 = count2 + 1
        
    file1 = '../data/2016/euclid_eigenvalue/' + name + '_hop' + str(hop) + '_median.csv'
    np.savetxt(file1 , matrix1 , delimiter = ',')        
    file2 = '../data/2016/euclid_eigenvalue/' + name + '_hop' + str(hop) + '_mean.csv'
    np.savetxt(file2 , matrix2 , delimiter = ',')
    
def attribute_to_file(start,end):
    for i in range(start,end):
        name = all_data[i]
        for hop in [1,2,3]:
            get_eigenvalue_median_and_mean(name,hop)


def pocket_test_feature(hop,typ):
    matrix = []
    for name in test_data:
        feature_file = '../data/2016/euclid_eigenvalue/' + name + '_hop'+ str(hop) +'_' + typ + '.csv'
        feature = np.loadtxt(feature_file,delimiter = ',')
        matrix.append(feature.tolist())
    feature_file = '../data/2016/euclid_feature/test_hop' + str(hop) + '_' + typ +'.csv'
    np.savetxt(feature_file , np.array(matrix) , delimiter = ',')
    print(np.array(matrix).shape)
    
    
def pocket_train_feature(hop,typ):
    matrix = []
    for name in train_data:
        feature_file = '../data/2016/euclid_eigenvalue/' + name + '_hop'+ str(hop) +'_' + typ + '.csv'
        feature = np.loadtxt(feature_file,delimiter = ',')
        matrix.append(feature.tolist())
    feature_file = '../data/2016/euclid_feature/train_hop' + str(hop) + '_' + typ +'.csv'
    np.savetxt(feature_file , np.array(matrix) , delimiter = ',')
    print(np.array(matrix).shape)   
    
def get_combined_feature(typ):
    feature_file = '../data/2016/euclid_feature/' + typ + 'hop1_median.csv'
    feature1 = np.loadtxt(feature_file,delimiter = ',')
    feature_file = '../data/2016/euclid_feature/' + typ + 'hop2_median.csv'
    feature2 = np.loadtxt(feature_file,delimiter = ',')
    feature_file = '../data/2016/euclid_feature/' + typ + 'hop3_median.csv'
    feature3 = np.loadtxt(feature_file,delimiter = ',')
    feature_file = '../data/2016/euclid_feature/' + typ + 'hop1_mean.csv'
    feature4 = np.loadtxt(feature_file,delimiter = ',')
    feature_file = '../data/2016/euclid_feature/' + typ + 'hop2_mean.csv'
    feature5 = np.loadtxt(feature_file,delimiter = ',')
    feature_file = '../data/2016/euclid_feature/' + typ + 'hop3_mean.csv'
    feature6 = np.loadtxt(feature_file,delimiter = ',')
    
    feature_hop1 = np.hstack((feature1,feature4))
    filename = '../data/2016/euclid_feature/' + typ + 'hop1_mm.csv'
    np.savetxt(filename,feature_hop1,delimiter=',')
    
    feature_hop2 = np.hstack((feature2,feature5))
    filename = '../data/2016/euclid_feature/' + typ + 'hop2_mm.csv'
    np.savetxt(filename,feature_hop2,delimiter=',')
    
    feature_hop3 = np.hstack((feature3,feature6))
    filename = '../data/2016/euclid_feature/' + typ + 'hop3_mm.csv'
    np.savetxt(filename,feature_hop3,delimiter=',')
    
    feature_hop12 = np.hstack((feature1,feature2,feature4,feature5))
    filename = '../data/2016/euclid_feature/' + typ + 'hop12_mm.csv'
    np.savetxt(filename,feature_hop12,delimiter=',')
    
    feature_hop123 = np.hstack((feature1,feature2,feature3,feature4,feature5,feature6))
    filename = '../data/2016/euclid_feature/' + typ + 'hop123_mm.csv'
    np.savetxt(filename,feature_hop123,delimiter=',')
    
def get_name_index(name,contents):
    t = len(contents)
    for i in range(t):
        if contents[i][0:4]==name:
            return i

def get_target_matrix_of_train():
    t = len(train_data)
    target_matrix = np.zeros(( t , 1 ))
    t1 = '../data/2016/refined-set/index/INDEX_refined_data.2016'
    f1 = open(t1,'r')
    contents = f1.readlines()
    f1.close()
    for i in range(t):  
        name = train_data[i]
        index = get_name_index(name,contents)
        target_matrix[i][0] = float(contents[index][18:23])
        
    target_file = '../data/2016/euclid_feature/train_target.csv'
    np.savetxt(target_file , target_matrix , delimiter = ',')  
       

def get_target_matrix_of_test():
    t = len(test_data)
    target_matrix = np.zeros(( t , 1 ))
    t1 = '../data/2016/refined-set/index/INDEX_refined_data.2016'
    f1 = open(t1,'r')
    contents = f1.readlines()
    f1.close()
    for i in range(t):  
        name = test_data[i]
        index = get_name_index(name,contents)
        target_matrix[i][0] = float(contents[index][18:23])
        
    target_file = '../data/2016/euclid_feature/test_target.csv'
    np.savetxt(target_file , target_matrix , delimiter = ',')  
    
# feature generation code ends
###########################################################################################################   



############################################################################################################
# machine_learning algorithm starts.

def gradient_boosting(X_train,Y_train,X_test,Y_test):
    params={'n_estimators': 40000, 'max_depth': 6, 'min_samples_split': 2,
                'learning_rate': 0.001, 'loss': 'ls','max_features':'sqrt','subsample':0.7}
    regr = GradientBoostingRegressor(**params)
    regr.fit(X_train,Y_train)
    pearson_coorelation = sp.stats.pearsonr(Y_test,regr.predict(X_test))
    mse = mean_squared_error(Y_test, regr.predict(X_test))
    rmse = pow(mse,0.5)
    return [pearson_coorelation[0],rmse]


def get_pearson_correlation(hop):
    feature_matrix_of_train = np.loadtxt( '../data/2016/euclid_feature/train_hop'+ str(hop) +'_mm.csv',delimiter=',' )
    target_matrix_of_train = np.loadtxt( '../data/2016/euclid_feature/train_target.csv',delimiter=',' )
    feature_matrix_of_test = np.loadtxt( '../data/2016/euclid_feature/test_hop'+ str(hop) +'_mm.csv',delimiter=',' )
    target_matrix_of_test = np.loadtxt( '../data/2016/euclid_feature/test_target.csv',delimiter=',' )
    number = 10
    P = np.zeros((number,1))
    M = np.zeros((number,1))
    print(feature_matrix_of_test.shape)
    print(feature_matrix_of_train.shape)
    print(target_matrix_of_test.shape,target_matrix_of_train.shape) 
    
    for i in range(number):
        [P[i][0],M[i][0]] = gradient_boosting(feature_matrix_of_train,target_matrix_of_train,feature_matrix_of_test,target_matrix_of_test)
        print(P[i])
    median_p = np.median(P)
    median_m = np.median(M)
    print('feature obtained by mean and median using hopping:' ,hop)
    print('for data 2016 , 10 results for euclid distance-model are:')
    print(P)
    print('median pearson correlation values are')
    print(median_p)
    print('median root mean squared error values are')
    print(median_m)   


# machine_learning algorithm ends. 
############################################################################################################



def run_PDBbind_2016():
    ###############################################################################
    '''
    run this function, you can get the results for data2016
    using mean and median attributes combined hopping 1,2,3
    '''
    ###############################################################################
    
    #obtain coordinate
    write_pocket_coordinate_to_file(0,1300,10)
    #nonzero eigenvalues to file
    eigenvalue_to_file(0,1300,10,0.1,3)
    #obtain associated median and mean
    attribute_to_file(0,1300)
    
    #pocket test/train feature
    for hop in [1,2,3]:
        for sta in ['median','mean']:
            pocket_test_feature(hop,sta)
            pocket_train_feature(hop,sta)
            
    #combine feature
    for typ in ['test','train']:    
        get_combined_feature(typ)
        
    #obtain target
    get_target_matrix_of_train()
    get_target_matrix_of_test()
    
    # machine learning
    get_pearson_correlation(1)
    get_pearson_correlation(2)
    get_pearson_correlation(3)
    get_pearson_correlation(12)
    get_pearson_correlation(123)
            
    
run_PDBbind_2016()
