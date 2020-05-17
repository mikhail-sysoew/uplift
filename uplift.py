#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Импорт нужных библиотек
import pandas as pd
import numpy as np
import warnings
import lightgbm
import causalml
import time
import matplotlib.pyplot as plt
import seaborn as sns


from causalml.propensity import ElasticNetPropensityModel
from causalml.match import NearestNeighborMatch, create_table_one
from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier, BaseRClassifier
from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor
from causalml.inference.meta import BaseRRegressor
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot

from scipy import stats
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from IPython.display import Image


# In[2]:


from causalml.propensity import ElasticNetPropensityModel
from causalml.match import NearestNeighborMatch, create_table_one
from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier, BaseRClassifier
from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor
from causalml.inference.meta import BaseRRegressor
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot

from scipy import stats
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from IPython.display import Image


class causalmodel():
    def __init__(self, model_type = 'random forest', feature_names=None, target_name=None, model_params=None):
        if model_type == 'random forest': self.model = UpliftRandomForestClassifier(n_estimators = model_params['n_estimators'],
                                                                                    max_features = model_params['max_features'],
                                                                                    random_state = model_params['random_state'],
                                                                                    max_depth = model_params['max_features'],
                                                                                    min_samples_leaf = model_params['min_samples_leaf'],
                                                                                    min_samples_treatment = model_params['min_samples_treatment'],
                                                                                    n_reg = model_params['n_reg'],
                                                                                    evaluationFunction = model_params['evaluationFunction'],
                                                                                    control_name = model_params['control_name'],
                                                                                    normalization = model_params['normalization']
                                                                                   )
        elif model_type == 'tree': self.model = UpliftTreeClassifier(
                                                                    max_depth = model_params['max_depth'],
                                                                    min_samples_leaf = model_params['min_samples_leaf'],
                                                                    min_samples_treatment = model_params['min_samples_treatment'],
                                                                    n_reg = model_params['n_reg'],
                                                                    evaluationFunction = model_params['evaluationFunction'],
                                                                    control_name = model_params['control_name']      
                                                                    ) 
        elif model_type == 's-learner': self.model = BaseSClassifier(LGBMClassifier(
                                                                    max_depth = model_params['max_depth'],
                                                                    reg_lambda = model_params['reg_lambda'],
                                                                    reg_alpha = model_params['reg_alpha'],
                                                                    n_estimators = model_params['n_estimators'],
                                                                    min_child_samples = model_params['min_child_samples'],
                                                                    num_leaves = model_params['num_leaves'],
                                                                                    ), 
                                                                     control_name='control')
        elif model_type == 't-learner': self.model = BaseTClassifier(LGBMClassifier(
                                                                    max_depth = model_params['max_depth'],
                                                                    reg_lambda = model_params['reg_lambda'],
                                                                    reg_alpha = model_params['reg_alpha'],
                                                                    n_estimators = model_params['n_estimators'],
                                                                    min_child_samples = model_params['min_child_samples'],
                                                                    num_leaves = model_params['num_leaves']
                                                                                    ), 
                                                                     control_name='control')            
        else: print("ERROR: No such model type")
        self.initial_df = [] #Initial dataframe        
        self.x = [] #vector x. Contains both control and treatment groups
        self.y = [] #vector target.
        self.treatment = [] #vector contains flag control and treatment group
        self.x_unbiased = [] #vector x after matching procedure
        self.y_unbiased = [] #vector y after matching procedure
        self.treatment_unbiased = [] #vector treatment after matching procedure
        self.feature_names = feature_names #features names of input
        self.feature_names_one_hot = [] #feature names after one-hot encoding
        self.flag_bias = False #Flag = True, if control group biased versus treatment group
        self.target_name = target_name #target column name
        self.model_type = model_type
        self.model_params = model_params
        self.df_unbiased=[]
      
    def check_type(self, column_name):
        #####################
        #INPUT - column name
        #OUTPUT - type of the column, categorial or numeric
        ########################
        from pandas.api.types import is_string_dtype
        from pandas.api.types import is_numeric_dtype   
        tmp_var = self.x[column_name][self.x[column_name].notnull()]
        #If number of uniques<=4 then type = categorial
        if tmp_var.nunique()<=4: return 'cat'
        elif is_numeric_dtype(tmp_var): return 'numeric'
        else: return 'cat'
        
        
    def preprocess(self, df):
        ###################
        #INPUT: initial dataframe
        #Procedure of data pre-processing.
        #Consist of 3 steps:             
        # 1. Delete raws, where target is null
        # 2. Fill empty values with -1 for numeric and MISSING for categorial
        # 3. Fill outliers with >3 standart deviations with border value
        # 4. Make one-hot  on categorial features
        # Write x,y and treatment into object from initial dataframe
        ####################
        
        #Step 1. Delete raws with target column name is null or treatment not calculated              
        #Assign x,y and treatment
        start_time = time.time()
        self.initial_df = df
        self.initial_df = self.initial_df[self.initial_df[self.target_name].notnull()]
        self.x = self.initial_df[self.feature_names]
        self.y = self.initial_df[self.target_name]
        self.treatment = self.initial_df['treatment']   
        
        #Step 2. Fill empty values on x
        for c in self.feature_names:
            if self.check_type(c)=='cat': self.x[c] = self.x[c].fillna('MISSING')
            if self.check_type(c)=='numeric': self.x[c] = self.x[c].fillna(-1)
        
        #Step 3. Fill outliers with >3 standart deviations with border value
        for c in self.feature_names:
            if self.check_type(c)=='numeric':                
                z = stats.zscore(self.x[c])    
                upper_bond = self.x[c][z>5].min()                
                lower_bond = self.x[c][z<-5].max()
                self.x[c] = self.x[c].apply(lambda x: min(x,upper_bond))
                self.x[c] = self.x[c].apply(lambda x: max(x,lower_bond))
                
        #Step 4. Make one-hot  on categorial features
        for c in self.feature_names:
            if self.check_type(c)=='cat': 
                self.x = self.x.merge(pd.get_dummies(self.x[c], prefix = c, drop_first=True), left_index=True, right_index=True)
                del self.x[c]
        self.feature_names_one_hot = self.x.columns
        print("Preprocess ends in: ", -start_time + time.time()," sec")
        
        
        
    def bias_elimination(self, caliper = None):
            print('-------Start unbias procedure----------')
            start_time = time.time()
            ############################
            # If control and target sample are biased, procedure make matching between them to eliminate this bias
            # Using Nearest Neigbors matching algorithm
            ############################
            #Join x, y and treatment vectors
            df_match = self.x.merge(self.treatment,left_index=True, right_index=True).replace({'control':1,'target':0})
            df_match = df_match.merge(self.y, left_index=True, right_index=True)
            
            #buld propensity model. Propensity is the probability of raw belongs to control group.
            pm = ElasticNetPropensityModel(n_fold=3, random_state=42)

            #ps - propensity score
            df_match['ps'] = pm.fit_predict(self.x, self.treatment)

            #Matching model object
            psm = NearestNeighborMatch(replace=False,
                           ratio=1,
                           random_state=423,
                           caliper=caliper)
            
            ps_cols = list(self.feature_names_one_hot)
            ps_cols.append('ps')
            
            #Apply matching model
            #If error, then sample is unbiased and we don't do anything
            self.flg_bias = True            
            self.df_unbiased = psm.match(data=df_match, treatment_col='treatment',score_cols=['ps'])                
            self.x_unbiased = self.df_unbiased[self.x.columns]
            self.y_unbiased = self.df_unbiased[self.target_name]
            self.treatment_unbiased = self.df_unbiased['treatment'].replace({1: 'control', 0: 'target'})
            print('-------------------MATCHING RESULTS----------------')
            print('-----BEFORE MATCHING-------')
            print(create_table_one(data=df_match,
                                    treatment_col='treatment',
                                    features=list(self.feature_names_one_hot)))        
            print('-----AFTER MATCHING-------')
            print(create_table_one(data=self.df_unbiased,
                                    treatment_col='treatment',
                                    features=list(self.feature_names_one_hot)))
        

    def fit_model(self):
            ##############
            #Collect preprocess, bias_elimination and fitting into one procedure           
          
            print("Start fitting uplift model")
            start_time = time.time()
            self.model.fit(self.x_unbiased.values,self.treatment_unbiased.values,self.y_unbiased.values)
            self.initial_df['y_pred'] = self.predict(self.x)
            print("Fitting uplift model ends in: ",-start_time + time.time()," sec")
            
            
    def predict(self,x):
            ###########
            #Standart predict method   
            if self.model_type=='tree': return self.model.predict(x.values)[1]
            else: return self.model.predict(x.values)
            
    def plot_tree(self):
            if self.model_type == 'tree':
                return uplift_tree_plot(self.model.fitted_uplift_tree,x_names = list(self.feature_names_one_hot.values))                
            else:
                print("ERROR: Model type should equal 'tree' for visualisation")
        
    def shift_effect(self):
        #return shift effect. Share of uplift, that can be explained by variables shifting.
        self.initial_df['y_pred'] = self.predict(self.x)
        uplift_effect = self.initial_df[self.initial_df['treatment']=='control']['y_pred'].mean() /  self.initial_df[self.initial_df['treatment']=='target']['y_pred'].mean()
        shift_effect = 1 - uplift_effect
        return shift_effect
    
    def plot_variable_uplift(self, var, var_type, bins_num, ntile, raw_data):
        ############PRINTING UPLIFT PLOT AND DISTRIBUTION#######################
        #print plot of uplift and variable distribution
        #INPUT:
        #var - name of the variable in dataframe
        #var_type - manually set of variable type: 'numeric' or 'categorial'
        #bins_num - number of bins in histogram (only if flg_raw=True)
        #ntile - number of ntiles (only for flg_raw=False)
        #raw_data - True print raw data, False ntile data
        #OUTPUT:
        #plot
        var_name = var
        
        if (var_type=='numeric') and raw_data==False:
            df_tmp = self.initial_df.copy()
            df_tmp[var] = df_tmp[var].fillna(0)        
            #df_tmp['y_pred'] = self.predict(self.x)
            df_control = df_tmp[df_tmp['treatment']=='control']
            df_target = df_tmp[df_tmp['treatment']=='target']          
            #df_tmp[var] = df_tmp[var].fillna(-10000)
            labels, bins = pd.qcut(df_control[var],ntile,retbins=True, precision=0, duplicates='drop')
            labels_nums = pd.qcut(df_control[var],ntile,labels=False, duplicates='drop')
            labels_names = labels_nums.astype(str).apply(lambda x: '0' if len(x)==1 else '')+ labels_nums.astype(str) +': '+ labels.astype(str)        
            df_tmp['var_bins'] = pd.cut(df_tmp[var], bins=bins,labels = sorted(labels_names.unique()), include_lowest=True).astype(str)
            df_tmp['zero']=0      

            sns.set_style("white")
            df_control = df_tmp[df_tmp['treatment']=='control']
            df_target = df_tmp[df_tmp['treatment']=='target']       

            var='var_bins'

            fig, ax = plt.subplots(figsize=(20,7))
            ax.hist(x=df_control.sort_values(by='var_bins')['var_bins'],histtype='bar',bins=ntile, alpha=0.5,rwidth=0.8, label='# distribution', density=True, color='grey', align='left')
            ax.hist(x=df_target.sort_values(by='var_bins')['var_bins'],histtype='bar', bins=ntile, alpha=0.3,rwidth=0.8, label='# distribution', density=True, color='green',align='left')
            ax2 =ax.twinx()
            #ax2.scatter(x = df_control[var],y=df_control['y_pred'],color='grey',alpha=0.5, s=1, label = 'Uplift Distribution')
            #ax2.plot(df_tmp['zero'])
            df_target.groupby(['var_bins']).mean()['y_pred'].plot(ax=ax2, color='black',lw=2, label = 'Average Uplift')
            df_control.groupby(['var_bins']).mean()['y_pred'].plot(ax=ax2, color='blue',lw=2, label = 'Average Uplift')
            df_tmp.groupby([var]).mean()['zero'].plot(ax=ax2, color='red',lw=2, label = 'Average Uplift')
            sns.despine(ax=ax, right=True, left=True)
            sns.despine(ax=ax2, left=True, right=False)
            ax2.spines['right'].set_color('grey')
            ax.set_title('Uplift by '+var_name)
            ax.legend(labels = ['Control Distribution','Target Distribution'],loc='upper left')
            ax2.legend(['Uplift with shift','Uplift without shift','Zero Uplift', 'Distibution of uplift'], loc='upper right')
            a = ax.set_xticklabels(labels=sorted(labels_names.unique()), rotation=90)
            plt.setp( ax.xaxis.get_majorticklabels(), rotation=90, ha='center') 
            #plt.setp( ax2.xaxis.get_majorticklabels(), rotation=90 ) 
        elif (var_type=='numeric') and raw_data:
            df_tmp = self.initial_df.copy()
            #df_tmp['y_pred'] = self.predict(self.x)
            df_tmp['zero']=0

            sns.set_style("white")
            df_control = df_tmp[df_tmp['treatment']=='control']
            df_target = df_tmp[df_tmp['treatment']=='target']

            #if raw_data: var=var
            #else: var = 'var_bins'

            fig, ax = plt.subplots(figsize=(20,7))
            ax.hist(x=df_control[var],histtype='bar',bins=bins_num, alpha=0.5,rwidth=0.8, label='# distribution', density=True, color='grey', align='mid')
            ax.hist(x=df_target[var],histtype='bar',bins=bins_num, alpha=0.3,rwidth=0.8, label='# distribution', density=True, color='green',align='mid')
            ax2 =ax.twinx()
            #ax2.scatter(x = df_control[var],y=df_control['y_pred'],color='grey',alpha=0.5, s=1, label = 'Uplift Distribution',align='center')
            #ax2.plot(df_tmp['zero'])
            df_target.groupby([var]).mean()['y_pred'].plot(ax=ax2, color='black',lw=2)
            df_control.groupby([var]).mean()['y_pred'].plot(ax=ax2, color='blue',lw=2)
            df_tmp.groupby([var]).mean()['zero'].plot(ax=ax2, color='red',lw=2)
            sns.despine(ax=ax, right=True, left=True)
            sns.despine(ax=ax2, left=True, right=False)
            ax2.spines['right'].set_color('grey')
            ax.set_title('Uplift by '+ var_name)
            ax.legend(labels = ['Control Distribution','Target Distribution'],loc='upper left')
            ax2.legend(['Uplift with shift','Uplift without shift','Zero Uplift', 'Distibution of uplift'], loc='upper right')
        
        elif (var_type=='cat'):
            df_tmp = self.initial_df.copy()            
            #df_tmp['y_pred'] = self.predict(self.x)
            df_tmp['zero']=0
            df_control = df_tmp[df_tmp['treatment']=='control']
            df_target = df_tmp[df_tmp['treatment']=='target']           
            sns.set_style("white")
            fig, ax = plt.subplots(figsize=(20,7))
            ax.hist(x=df_control.sort_values(by=var)[var],histtype='bar', alpha=0.5,rwidth=0.8, label='# distribution', density=True, color='grey', align='left')
            ax.hist(x=df_target.sort_values(by=var)[var],histtype='bar', alpha=0.3,rwidth=0.8, label='# distribution', density=True, color='green',align='left')
            ax2 =ax.twinx()
            df_target.groupby([var]).mean()['y_pred'].plot(ax=ax2, color='black',lw=2)
            df_control.groupby([var]).mean()['y_pred'].plot(ax=ax2, color='blue',lw=2)
            df_tmp.groupby([var]).mean()['zero'].plot(ax=ax2, color='red',lw=2)
            sns.despine(ax=ax, right=True, left=True)
            sns.despine(ax=ax2, left=True, right=False)
            ax2.spines['right'].set_color('grey')
            ax.set_title('Uplift by '+ var_name)
            ax.legend(labels = ['Control Distribution','Target Distribution'],loc='upper left')
            ax2.legend(['Uplift with shift','Uplift without shift','Zero Uplift', 'Distibution of uplift'], loc='upper right')
            
    def plot_total_uplift(self,quantiles):
        df_t = self.initial_df.copy()
        df_t['y_pred_bins'] = pd.qcut(df_t['y_pred'],q=quantiles, precision=1)
        df_t['y'] = df_t[self.target_name]

        dy_table = pd.DataFrame([])
        dy_table['y_control'] = df_t[df_t['treatment']=='control'].groupby('y_pred_bins')['y'].mean()
        dy_table['labels'] = dy_table.index.values
        dy_table['y_target'] = df_t[df_t['treatment']=='target'].groupby('y_pred_bins')['y'].mean()
        dy_table['y_pred'] = df_t[df_t['treatment']=='target'].groupby('y_pred_bins')['y_pred'].mean()
        dy_table['cnt_control'] = df_t[df_t['treatment']=='control'].groupby('y_pred_bins')['y'].count()
        dy_table['cnt_target'] = df_t[df_t['treatment']=='target'].groupby('y_pred_bins')['y'].count()
        dy_table['zero']=0

        dy_table['dy'] = dy_table['y_target'] - dy_table['y_control']
        l = [i for i in range((len(dy_table)))]
        dy_table['num'] = l
        dy_table

        fig, ax = plt.subplots(figsize=(10,7.5))

        ax = sns.lineplot(x="num", y="dy",data=dy_table)
        ax = sns.lineplot(x="num", y="y_pred",data=dy_table,sort=True)
        ax = sns.lineplot(x="num", y="zero",data=dy_table,sort=True, color='black', lw=2)
        ax.set_title('Predicted uplift VS Factual change')
        ax.set_xlabel('Equal parts of sample')
        ax.set_ylabel('Absolute change of target')
        ax.legend(['Factual change', 'Predicted change', 'Zero uplift'])
        
    def plot_uplift_distribution(self):
        fig, ax = plt.subplots(figsize=(7,5))
        print("Average uplift on control: %5.4f"%(self.initial_df[self.initial_df['treatment']=='control']['y_pred'].mean()))
        print("Average uplift on control: %5.4f"%(self.initial_df[self.initial_df['treatment']=='target']['y_pred'].mean()))
        ax = sns.distplot(self.initial_df[self.initial_df['treatment']=='control']['y_pred'])
        ax = sns.distplot(self.initial_df[self.initial_df['treatment']=='target']['y_pred'])
        ax.set_title('Distribution by uplift')
        ax.set_xlabel('Predicted uplift')
        ax.legend(['Control','Target'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    def plot_feature_importances(self):
        self.model.plot_importance(X=self.x, tau=self.initial_df['y_pred'], normalize=True, method='auto',features = m.feature_names_one_hot)

