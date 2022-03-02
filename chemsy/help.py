from chemsy.prep.methods import *
from chemsy.predict.methods import *
from inspect import getmembers, isclass


def see_methods(module):
    class_members=getmembers(module, isclass)
    search_list=[]
    for x in class_members:
        search_list.append(x[0])
    delete_list=['C','RBF','ShuffleSplit','TransformerMixin','RegressorMixin','BaselineRemoval','BaseEstimator']    
    output_list= [x for x in search_list if x not in delete_list]
    print('Method supported:')
    print(output_list)
    