'''
Assume df is a pandas dataframe object of the dataset given
'''
import math
import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    # TODO
    col=df.iloc[:,-1]
    count=list(col.value_counts())
    probabilities = [i/len(col) for i in count]
    entropy = 0
    for p in probabilities:
        if p> 0:
            entropy -= p* math.log(p, 2)
    return entropy

'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    values = df[attribute].unique()
    attribute_entropy=0
    for value in values:
        subset = df[df[attribute] == value]
        counts=subset[[subset.columns[-1]]].value_counts()
        total_count = np.sum(counts)
        entropy = 0
        probabilities=[i/total_count for i in counts]
        for p in probabilities:
            if p>0:
                entropy-=p*math.log(p,2)
        attribute_entropy+=entropy*(total_count/df.shape[0])
    avg_info=abs(attribute_entropy)
    return avg_info

'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    information_gain=0
    # TODO
    information_gain=get_entropy_of_dataset(df)-get_avg_info_of_attribute(df,attribute)
    return information_gain

#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    # TODO

    info_gain={}
    max_col=""
    max_ig=0
    for col in df.columns[:-1]:
        ig_col=get_information_gain(df,col)
        if ig_col>max_ig:
            max_col=col
            max_ig=ig_col
        info_gain[col]=ig_col
    return (info_gain,max_col)


