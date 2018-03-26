from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

class CalculateQuantativeFeatures(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""
    from scipy.sparse import csr_matrix
    def __init__(self):
        pass

    def compute_product_aggregates(self, df):
        '''Computes aggregates for each 'ProductId' Returns
        data frame with additional 2 columns for 'minTimePerProduct' 
        and 'numReviewsPerProduct'''
        #define aggregations on 'ProductId' group
        aggregations = {'Time': 'min','Id': "count"}
        
        #create groupBy object on 'ProductId' and preform aggregations
        #'as_index = False' so we have the 'ProductId' column to join on later
        productGrouped = df.groupby(['ProductId'], as_index= False).agg(aggregations)
        
        # rename columns
        productGrouped = productGrouped.rename(columns={"Time": "minTimePerProduct",'Id': 'numReviewsPerProduct'})
        
        # add merged values to original data
        merged = pd.merge(df, productGrouped, how = 'left', on = ['ProductId'])
        
        merged['timeDiffFromFirstReview'] = merged["Time"]-merged["minTimePerProduct"]
        
        return merged
    
    def transform(self, df):
        '''The workhorse of this feature extractor'''
        # calculates raw data merged with product aggregates
        quantFeatures = self.compute_product_aggregates(df)
        
        #create column for str length of text
        quantFeatures['reviewLen'] = quantFeatures['Text'].str.len()
        
        X_quant_features = quantFeatures[["Score", 
                                          "reviewLen", 
                                          "minTimePerProduct", 
                                          "numReviewsPerProduct", 
                                          "timeDiffFromFirstReview"]]
        
        #convert to sparce matric so can be combined with other features
        X_quant_features_csr = csr_matrix(X_quant_features)
        return X_quant_features_csr

    def fit(self, df, y=None):
        '''Returns `self` unless something different happens in train and test'''
        return self
    
class SelectCol(BaseEstimator, TransformerMixin):

    def __init__(self, col='Text'):
        self.col = col

    def transform(self, X):
        if self.col == 'Summary':
            return X[self.col].values.astype('str')
        else:
            return X[self.col]

    def fit(self, X, y=None):
        return self
