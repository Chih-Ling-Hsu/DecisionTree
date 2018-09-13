import sys
import pandas as pd
import numpy as np
import copy
import math
from pandas.api.types import is_numeric_dtype
import numbers

class DecisionTree():
    def __init__(self, method="ID3", measure="entropy", max_depth=5):
        self.root = None
        self.method = method
        self.measure = measure
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self.buildTree(X, y, list(X.columns.values), self.max_depth)        

    def predict(self, X):
        y = []
        for index, row in X.iterrows():
            y.append(self.traverseTree(self.root, row))
        return y

    def traverseTree(self, node, X):
        if node.isLeaf==True:
            return node.label
        else:
            f = X[node.feature]
            if node.split==None:
                return self.traverseTree(node.branch[f], X)
            else:
                if f < node.split:
                    return self.traverseTree(node.branch["<"], X)
                else:
                    return self.traverseTree(node.branch[">="], X)

    def exportTree(self):
        self.showTree(self.root, 0)

    def showTree(self, node, indent):
        if node.isLeaf==True:
            print("    "*indent +"===> " + node.label)
            return
        
        print("    "*indent +"--- "+ "[" + node.feature + "]")
        indent += 1        
        for condition in node.branch:
            if node.split==None:
                print("    "*indent +"--- "+ condition)
            else:
                print("    "*indent +"--- "+ condition + str(node.split))
            self.showTree(node.branch[condition], indent+1)

    def buildTree(self, X, y, attributes, depth_remain):
        if self.method=="C4.5":
            return self.C4dot5(X, y, attributes, depth_remain)
        else:
            return self.ID3(X, y, attributes, depth_remain)
    

    '''
    ID3 (Examples, Target_Attribute, Attributes)
    Create a root node for the tree
    If all examples are positive, Return the single-node tree Root, with label = +.
    If all examples are negative, Return the single-node tree Root, with label = -.
    If number of predicting attributes is empty, then Return the single node tree Root,
    with label = most common value of the target attribute in the examples.
    Otherwise Begin
        A ← The Attribute that best classifies examples.
        Decision Tree attribute for Root = A.
        For each possible value, vi, of A,
            Add a new tree branch below Root, corresponding to the test A = vi.
            Let Examples(vi) be the subset of examples that have the value vi for A
            If Examples(vi) is empty
                Then below this new branch add a leaf node with label = most common target value in the examples
            Else below this new branch add the subtree ID3 (Examples(vi), Target_Attribute, Attributes – {A})
    End
    Return Root
    '''
    def ID3(self, X, y, attributes, depth_remain):
        root = TreeNode()
        unique_vals, counts = np.unique(y.values, return_counts=True)
        if len(unique_vals) == 1:
            root.setLabel(unique_vals[0])
        elif len(attributes)==0 or depth_remain == 0:
            root.setLabel(unique_vals[np.argmax(counts)])
        else:
            attr, split_criterion = self.findBestAttribute(X, y, attributes)
            new_attributes = copy.deepcopy(attributes)
            new_attributes.remove(attr)
            root.setFeature(attr)
            unique_vals = np.unique(X[attr].values)
            for val in unique_vals:
                root.addBranch(self.ID3(X[X[attr] == val], y[X[attr] == val], new_attributes, depth_remain-1), val)
        return root

    def C4dot5(self, X, y, attributes, depth_remain):
        root = TreeNode()
        unique_vals, counts = np.unique(y.values, return_counts=True)
        if len(unique_vals) == 1:
            root.setLabel(unique_vals[0])
        elif len(attributes)==0 or depth_remain == 0:
            root.setLabel(unique_vals[np.argmax(counts)])
        else:
            attr, split_criterion = self.findBestAttribute(X, y, attributes)
            #print(attr, split_criterion)
            if attr == None:
                root.setLabel(unique_vals[np.argmax(counts)])
            else:
                new_attributes = copy.deepcopy(attributes)
                new_attributes.remove(attr)
                root.setFeature(attr)
                if split_criterion==None:
                    unique_vals = np.unique(X[attr].values)
                    for val in unique_vals:
                        root.addBranch(self.C4dot5(X[X[attr] == val], y[X[attr] == val], new_attributes, depth_remain-1), val)
                else:
                    Z = X[attr].values < split_criterion
                    root.addBranch(self.C4dot5(X[Z], y[Z], new_attributes, depth_remain-1), "<", split_criterion)
                    root.addBranch(self.C4dot5(X[~Z], y[~Z], new_attributes, depth_remain-1), ">=", split_criterion)
        return root


    def findBestAttribute(self, X, y, attributes):
        optimal_attr = None
        optimal_split = None
        maxIG = 0
        for attr in attributes:
            #print("[["+attr+"]]")
            IG, split_criterion = self.gain(X, y, attr)
            #print(IG)
            if optimal_attr == None or IG > maxIG:
                optimal_attr = attr
                optimal_split = split_criterion
                maxIG = IG
        if maxIG == 0:
            optimal_attr = None
            optimal_split = None
        #print("-------->"+optimal_attr+"\n")
        return optimal_attr, optimal_split

    
    def gain(self, X, y, attr):
        if self.method=="C4.5":
            if is_numeric_dtype(X[attr]):
                IG, split_criterion = self.findBestSplit(X, y, attr)
            else:
                IG = gain_ratio(X, y, attr)
                split_criterion = None
        else:
            IG = information_gain(X, y, attr)
            split_criterion = None
        return IG, split_criterion

    def findBestSplit(self, X, y, attr):
        valList = np.sort(X[attr].values)

        optimal_split = None
        maxIG = 0
        for idx in range(len(valList)-1):
            if valList[idx]==valList[idx+1]:
                continue
            split_criterion = float(valList[idx] + valList[idx+1])/2
            #print("split at:", split_criterion)
            IG = gain_ratio(X, y, attr, split_criterion)
            #print("= ", IG)
            if optimal_split == None or IG > maxIG:
                optimal_split = split_criterion
                maxIG = IG    
            
        #print("--->", maxIG, optimal_split)
        return maxIG, optimal_split

def entropy(y):
    n = y.shape[0]
    unique_vals, counts = np.unique(y.values, return_counts=True)
    e = 0
    for nt in counts:  
        prob = float(nt/n)
        e += -prob * math.log(prob, 2)
    #print("   +"+str(e))
    return e

def information_gain(X, y, attr):
    unique_vals, counts = np.unique(X[attr].values, return_counts=True)
    #print("IG: ")
    IG = entropy(y)
    n = X.shape[0]
    for (val, nt) in zip(unique_vals, counts):        
        #print(val)
        child = y[X[attr] == val]
        #print("(*"+str(nt)+"/"+str(n)+")")
        IG -= (nt/n) * entropy(child)
    #print("  = "+str(IG))
    return IG

def gain_ratio(X, y, attr, split_criterion = None):
    if split_criterion==None:
        unique_vals, counts = np.unique(X[attr].values, return_counts=True)   
        IG = entropy(y)
        n = X.shape[0]
        for (val, nt) in zip(unique_vals, counts):      
            child = y[X[attr] == val]
            IG -= (nt/n) * entropy(child)
    else:
        Z = X[attr].values < split_criterion
        unique_vals, counts = [True, False], [X[Z].shape[0], X[~Z].shape[0]]
        if 0 in counts: 
            counts.remove(0)
        IG = entropy(y)
        #print("ori:"+str(IG))
        n = X.shape[0]
        for (val, nt) in zip(unique_vals, counts):      
            if val==True: child = y[Z]
            else: child = y[~Z]
            #print(child)
            #print("(*"+str(nt)+"/"+str(n)+")")
            IG -= (nt/n) * entropy(child)
        #print("="+str(IG))

    split_info = 0
    for nt in counts:  
        prob = float(nt/n)
        split_info += -prob * math.log(prob, 2)
    if len(counts) == 1:
        return IG
    else:
        #print(IG, "/", split_info)
        return IG/split_info

class TreeNode():
    def __init__(self):
        self.feature = ""
        self.label = ""
        self.isLeaf = False
        self.branch = {}
        self.split = None

    def setFeature(self, feature):
        self.feature = feature

    def setLabel(self, label):
        self.label = label
        self.isLeaf = True

    def addBranch(self, child, condition, split=None):
        self.split = split
        self.branch[condition] = child
    

if __name__ == "__main__":
    inputPath = sys.argv[1]
    target = sys.argv[2]

    #df = pd.read_csv(inputPath, header=0, index_col=False)
    df = pd.read_csv(inputPath, header=0, index_col=False, sep='\s+')
    '''del df["customer_id"] #df["customer_id"] = df["customer_id"].astype('str')
    del df["account_num"] #df["account_num"] = df["account_num"].astype('str')
    del df["postal_code"] #df["postal_code"] = df["postal_code"].astype('str')
    del df["phone"]
    df["customer_region_id"] = df["customer_region_id"].astype('str')'''
    y = df[target]
    X = copy.deepcopy(df)
    del X[target]

    dt = DecisionTree(method="C4.5", max_depth=3)
    dt.fit(X, y)
    dt.exportTree()

    #print(dt.predict(X))

