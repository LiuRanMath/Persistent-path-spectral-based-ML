Persistent path spectral (PPS) based machine learning for protein-ligand interactions
====
    This manual is for the code implementation of paper "Persistent path spectral (PPs) based machine learning for protein-ligand binding affinity prediction"
    
****

# Software configuration
---
        Platform: Python>=3.6
        Packages needed: math, numpy>=1.18.1, scipy>=1.4.1, scikit-learn>=0.22.1

# Flowchart of PPS-ML model
---
Protein-ligand complex  -->  PPS based representation  -->   Feature generation  -->  Machine learning 
# Details about each step
We take the PDBbind-v2007 and our PPS-ML(Dist) model as an example to show how to reproduce our results.  
## Protein-ligand complex
Before the representatoin, we need to get the protein-ligand complex data, which can be downloaded from the PDBbind website www.pdbbind.org.cn. Then the following function can be used to extract the 3D-coordinates of the protein-ligand complex.
```python
def write_pocket_coordinate_to_file(start,end,cutoff)
    # start and end are set to be 0,1300; 0,2959; 0,4057 for PDBbind-v2007, PDBbind-v2013 and PDBbind-v2016 respectively
    # cutoff is set to be 10, which is used to extract the binding core region

```
The above function can be found in "code/euclid2007.py"

## PPS based representatoin
For each protein-ligand complex, 36 element-specific atom-combinations are generated as in the paper. For each atom combination, a filtered bipartite graph is generated. Then, persistent path-spectral in dimension 0 with hopping 1, 2 and 3 are calculated.
The following function can be used to do this
```python
def eigenvalue_to_file(start,end,max_filtration,step,k)
    # start, end are the starting and ending index of the data, default are 0,1300; 0,2959; 0,4057 for PDBbind-v2007, PDBbind-v2013 and PDBbind-v2016 respectively.
    # max_filtration is set to be 10, which is the filtratoin value
    # step is set to be 0.1
    # k is set to be 3, which means we calculeta 1-hopping, 2-hopping and 3-hopping

```
The above function can be found in "code/euclid2007.py"

## Feature generation
Persistent mean value and persistent median value are used as features in our model. The following function can be used to generate the features
```python
def attribute_to_file(start,end)
    # start, end are the starting and ending index of the data, default are 0,1300; 0,2959; 0,4057 for PDBbind-v2007, PDBbind-v2013 and PDBbind-v2016 respectively.
    

```
The above function can be found in "code/euclid2007.py"

## Machine learning
GradientBoostingRegressor is used to do the regression. The following function can be used to make the regression
```python
def gradient_boosting(X_train,Y_train,X_test,Y_test)
    # X_train is the feature of training set
    # Y_train is the label of training set
    # X_test is the feature of test set
    # Y_test is the label of test set

```
The above function can be found in "code/euclid2007.py"

## A easier way to reproduce our results
We have provided the first ten entries of PDBbind-v2007. You can firstly download the data from PDBbind website and put all 1300 entries in the directory "data/2007/refined-set/", then you can directly run the script "code/euclid2007.py" to get our results for PDBbind-v2007.
