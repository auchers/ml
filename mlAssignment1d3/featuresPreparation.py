import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler

class AmazonReviewFeaturePrep():
    '''Procedure for creating features for Amazon Review Dataset'''
    
    def __init__(self, Xdata):
        '''Initialize attributes: training data'''
        self.rawData = Xdata
        self.X_hv = False
        self.X_tfidf = False
        self.summary_tfidf = False
        self.X_quantFeatures = False
        self.X = False #final X for training
    
    def compute_all_features_train(self):
        '''Compute all features available in module, fitting them to the TRAINING set'''
        '''Includes Product Aggregates (minTime, numReviews), Hash Vectorizor,  TDIF, String Length'''
        print('starting to compute features (out of 14)')
        self.compute_tfidf()
        self.compute_summary_text_stats() ##NEW for SUMMARY
        self.compute_quant_features()
        #combine two space matrixes by concatinating them
        X_combined = hstack([self.X_tfidf, self.summary_tfidf, self.X_quantFeatures])
        # convert to sparse matrix
        X_matrix = csr_matrix(X_combined) 
        
        # scale resulting matrix
        self.X = self.scale_training_matrix(X_matrix)
        print('14. finished computing all features')
        
    def prepare_all_features_test(self):
        '''Compute all features available in module, applying the fit from the training set pickle files to TEST set'''
        print('starting to prepare features')
        self.prepare_tfidf()
        
        #add summary_tfidf()
        self.prepare_summary_text_stats()
        
        self.compute_quant_features()
        #combine two space matrixes by concatinating them
        X_combined = hstack([self.X_tfidf, self.summary_tfidf, self.X_quantFeatures])
        # convert to sparse matrix
        X_matrix = csr_matrix(X_combined)
        
        #apply scale to resulting matrix
        self.X = self.scale_test_matrix(X_matrix)
        print('finished preparing test features')

    def prepare_hash_vectorizor(self):
        '''Applies hash vectorizor to test data'''
        hv = joblib.load('hv.pkl')
        self.X_hv = hv.transform(self.rawData.Text)
        print('1. finished applying hash vectorizor')
    
    def prepare_tfidf(self):
        '''Applies tfidf transformer to test data'''
        # load and apply hash vectorizor
        self.prepare_hash_vectorizor()
        # load tfidf transformer
        transformer = joblib.load('transformer.pkl')
        # apply transformer to hash vectorizor
        self.X_tfidf = transformer.transform(self.X_hv)
        print('2. finished applying tfidf')
        
    def prepare_summary_text_stats(self):
        '''Applies transformers to summary text data'''
        # load hash vectorizor
        sum_hv = joblib.load('summary_hv.pkl')
        summary_hv = sum_hv.transform(self.rawData.Text)
        # load tfidf transformer
        transformer = joblib.load('summary_transformer.pkl')
        # apply transformer to hash vectorizor
        self.summary_tfidf = transformer.transform(summary_hv)
        print('3. finished applying summary text transformers')
    
    def compute_hash_vectorizor(self):
        '''Computes vectorize Bag of Words from review text; as sparse matrix 
        and saves Pickle File of hash named 'hv.pkl' '''
        print('1. starting Hash Vectorizor')
        #initialize Hashing Vectorizer
        hv = HashingVectorizer(n_features=2 ** 17, non_negative=True)
        #fits vectorizor to data
        self.X_hv = hv.fit_transform(self.rawData.Text)
        #saves model fit
        joblib.dump(hv, 'hv.pkl')
        print('2. hv.pkl saved')
        
    def compute_tfidf(self):
        '''Computes tdif transformation on hash vectorizor; as sparse matrix 
        and saves Pickle File of hash named 'transformer.pkl' '''
        print('3. starting tfidf')
        #compute hash vectorizor
        self.compute_hash_vectorizor()
        #initialize Tfidf transformer
        transformer = TfidfTransformer()
        #fit transformer to model
        self.X_tfidf = transformer.fit_transform(self.X_hv)
        #saves transformers fit
        joblib.dump(transformer, 'transformer.pkl')
        print('4. transformer.pkl saved')
        
    def compute_summary_text_stats(self):
        '''Computes hash vectorizor and tfidf on summary text'''
        print('5. starting Summary Text Stats')
        # first, compute initialize new hashing vectorizor
        hv = HashingVectorizer(n_features=2 ** 17, non_negative=True)
        # fit to data
        summary_hv = hv.fit_transform(self.rawData
                                      .Summary.values.astype('str'))
        # save model fit
        joblib.dump(hv, 'summary_hv.pkl')
        print('6. summary_hv.pkl saved')
        
        # summary tfidf
        transformer = TfidfTransformer()
        self.summary_tfidf = transformer.fit_transform(summary_hv)
        
        #save model fit
        joblib.dump(transformer, 'summary_transformer.pkl')
        print('7. summary_transformer.pkl saved')
    
    def compute_quant_features(self):
        '''Computes string length, score, and product aggregations (minTimePerProduct and numReviewsPerProduct) then transforms to sparse matrix '''
        print('8. starting quant features')
        # calculates raw data merged with product aggregates
        quantFeatures = self.compute_product_aggregates()
        #create column for str length of text
        quantFeatures['reviewLen'] = quantFeatures['Text'].str.len()
        #select columns for training
        #TODO: think about removing "minTimePerProduct"
        X_quant_features = quantFeatures[["Score", "reviewLen", "minTimePerProduct", "numReviewsPerProduct", "timeDiffFromFirstReview"]]
        #conver to sparce matric so can be combined with other features
        X_quant_features_csr = csr_matrix(X_quant_features)
        self.X_quantFeatures = X_quant_features_csr
        print('11. finished quant features')
    
    def compute_product_aggregates(self):
        '''Computes aggregates for each 'ProductId'
        Returns data frame with additional 2 columns for 'minTimePerProduct' and 'numReviewsPerProduct'''
        print('9. starting product aggregates')
        #define aggregations on 'ProductId' group
        aggregations = {'Time': 'min','Id': "count"}
        #create groupBy object on 'ProductId' and preform aggregations
        #'as_index = False' so we have the 'ProductId' column to join on later
        productGrouped = self.rawData.groupby(['ProductId'], as_index= False).agg(aggregations)
        # rename columns
        productGrouped = productGrouped.rename(columns={"Time": "minTimePerProduct",'Id': 'numReviewsPerProduct'})
        # add merged values to original data
        merged = pd.merge(self.rawData, productGrouped, how = 'left', on = ['ProductId'])
        # calculate the difference btwn current review time to the first for that product
        merged['timeDiffFromFirstReview'] = merged["Time"]-merged["minTimePerProduct"]
        print('10. finished product aggregates')
        return merged
        
    def scale_training_matrix(self, matrix):
        ''' Scales concatinated feature matrix and saves fit to pickle file called 'sc.pkl' '''
        print('12. starting matrix scaling')
        #initialize scaler
        sc = StandardScaler(with_mean=False)
        X = sc.fit_transform(matrix)
        joblib.dump(sc, 'sc.pkl') # pickle
        print('13. sc.pkl saved')
        return X
    
    def scale_test_matrix(self, matrix):
        # load scale from training set
        sc = joblib.load('sc.pkl')
        # apply it to new x matrix
        X = sc.transform(matrix)
        print('scale applied')
        return X
        