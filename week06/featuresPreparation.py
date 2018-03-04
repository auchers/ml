import numpy as np
import pandas as pd

class AmazonReviewFeaturePrep():
    '''Procedure for creating features for Amazon Review Dataset'''
    
    def __init__(self, trainingXdata):
        '''Initialize attributes: training data'''
        self.data = trainingXdata
    
    def compute_all_features(self):
        '''Compute all features available in module'''
        '''Includes Product Aggregates (minTime, numReviews), Hash Vectorizor,  TDIF, String Length'''
        self.data = self.compute_product_aggregates()
    
    def compute_product_aggregates(self):
        '''Computes aggregates for each 'ProductId'
        Returns data frame with additional 2 columns for 'minTimePerProduct' and 'numReviewsPerProduct'''
        
        #define aggregations on 'ProductId' group
        aggregations = {'Time': 'min','Id': "count"}
        
        #create groupBy object on 'ProductId' and preform aggregations
        #'as_index = False' so we have the 'ProductId' column to join on later
        productGrouped = self.data.groupby(['ProductId'], as_index= False).agg(aggregations)

        # rename columns
        productGrouped = productGrouped.rename(columns={"Time": "minTimePerProduct",'Id': 'numReviewsPerProduct'})
        
        # add merged values to original data
        merged = pd.merge(self.data, productGrouped, how = 'left', on = ['ProductId'])
        return merged
    
    def compute_measures(self):
        '''Compute performance measures defined by Flach p. 57'''
        self.performance_measures['Pos'] = self.performance_df['labls'].sum()
        self.performance_measures['Neg'] = self.performance_df.shape[0] - self.performance_df['labls'].sum()
        self.performance_measures['TP'] = ((self.performance_df['preds'] == True) & (self.performance_df['labls'] == True)).sum()
        self.performance_measures['TN'] = ((self.performance_df['preds'] == False) & (self.performance_df['labls'] == False)).sum()
        self.performance_measures['FP'] = ((self.performance_df['preds'] == True) & (self.performance_df['labls'] == False)).sum()
        self.performance_measures['FN'] = ((self.performance_df['preds'] == False) & (self.performance_df['labls'] == True)).sum()
        self.performance_measures['Accuracy'] = (self.performance_measures['TP'] + self.performance_measures['TN']) / (self.performance_measures['Pos'] + self.performance_measures['Neg'])
        self.performance_measures['Precision'] = self.performance_measures['TP'] / (self.performance_measures['TP'] + self.performance_measures['FP'])
        self.performance_measures['Recall'] = self.performance_measures['TP'] / self.performance_measures['Pos']
        self.performance_measures['desc'] = self.desc

    def img_indices(self):
        '''Get the indices of true and false positives to be able to locate the corresponding images in a list of image names'''
        self.performance_df['tp_ind'] = ((self.performance_df['preds'] == True) & (self.performance_df['labls'] == True))
        self.performance_df['fp_ind'] = ((self.performance_df['preds'] == True) & (self.performance_df['labls'] == False))
        self.image_indices['TP_indices'] = np.where(self.performance_df['tp_ind']==True)[0].tolist()
        self.image_indices['FP_indices'] = np.where(self.performance_df['fp_ind']==True)[0].tolist()