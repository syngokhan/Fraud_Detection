#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.set_option("display.max_columns",None)
pd.set_option("display.float_format", lambda x : "%.4f" %x)
pd.set_option("display.width" , 200)


# In[3]:


from warnings import filterwarnings
import gc
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore",category=FutureWarning)
filterwarnings("ignore")


# In[4]:


get_ipython().run_cell_magic('time', '', '\npath = "/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data"\n\ntrain = pd.read_csv(f"{path}/drop_train.csv", index_col=0)\ntest  = pd.read_csv(f"{path}/drop_test.csv" , index_col=0)\n\ngc.collect()')


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


X = train.drop(["TransactionID","isFraud"], axis = 1) 
y = train["isFraud"]


# In[8]:


train_cat_cols = pd.read_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data/train_cats.csv",
                            index_col=0)
test_cat_cols = pd.read_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data/test_cats.csv",
                            index_col=0)


# In[9]:


train_cat_cols.shape,test_cat_cols.shape


# In[10]:


(train_cat_cols.replace("_","",regex = True) == test_cat_cols.replace("-","",regex = True).                                                replace("_","",regex = True)).sum().values[0]


# In[11]:


train_cat_cols = train_cat_cols.get("0").values.tolist()
test_cat_cols = test_cat_cols.get("0").values.tolist()


# In[12]:


train_num_cols = [col for col in X.columns if col not in train_cat_cols]
train_num_cols[:20]


# In[13]:


# Test Uygulamadık !!!!

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_Scaled = sc.fit_transform(X[train_num_cols])


# In[14]:


X_Scaled = pd.DataFrame(data = X_Scaled, columns = train_num_cols)
X_Scaled = pd.concat([ X_Scaled, X[train_cat_cols] ], axis = 1)
X_Scaled.head()


# In[15]:


print("X_Scaled Shape : {}".format(X_Scaled.shape))
print("X Shape : {}".format(X.shape))


# # Model

# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score,roc_curve,f1_score,precision_score,accuracy_score,recall_score,                            confusion_matrix, classification_report

from sklearn.model_selection import train_test_split,cross_validate,validation_curve
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold


# In[17]:


######################################################
# Automated Hyperparameter Optimization
######################################################


xgboost_params = {
                   "learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 0.8, 1]}



rf_params = {"max_depth": [ 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [ 500, 1000]
            }

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

#"criterion" : ["gini","entropy"],
#"splitter" : ["best","random"],
cart_params = {
                'max_depth': np.arange(1,20,5,dtype = int),
                "min_samples_split": np.arange(2,30,5,dtype = int)}



# "C" : [100, 10, 1.0, 0.1, 0.01],
#"penalty" : ['l1', 'l2', 'elasticnet', 'none'],
#"solver" : ['newton-cg', 'lbfgs', 'liblinear'],

lr_params = {
             "C" : [1.0, 0.1, 0.01],
             "max_iter" : np.linspace(100,300,3,dtype = int)
             }


lightgbm_params = {
                    "learning_rate" : [0.1, 0.2],
                    #"max_depth" : [5,8,12],
                    "n_estimators" : [100,200,300],
                    "colsample_bytree" : [0.5, 1]
                  }

classifiers = [ 
                      
                       ("RF", RandomForestClassifier(), rf_params),
                       ("CART", DecisionTreeClassifier() , cart_params),    
                       ("LR" , LogisticRegression() , lr_params),
                       ("GBM" , GradientBoostingClassifier() , gbm_params),
                       ("XGBoost" , XGBClassifier(), xgboost_params),
                       ("LightGBM", LGBMClassifier() , lightgbm_params),
]


# In[18]:


def base_advanced_model(classifiers, X , y , cv =5 ):
    
    base_data = pd.DataFrame()
    base_dict = {}
    base_index = 0
    
    
    for name, classifier, params in classifiers:
        
        cv_results = cross_validate(estimator = classifier,
                                    X = X,
                                    y = y,
                                    cv = cv,
                                    n_jobs=-1, 
                                    verbose = 0 ,
                                    scoring = ["roc_auc","accuracy"])
    
    
        base_roc_auc = cv_results["test_roc_auc"].mean()
        base_accuracy = cv_results["test_accuracy"].mean()
        base_fit_time = cv_results["fit_time"].mean()
        base_score_time = cv_results["score_time"].mean()
        
        base_data.loc[base_index, "NAME"] = name
        base_data.loc[base_index, "ROC_AUC_SCORE"] = base_roc_auc
        base_data.loc[base_index, "ACCURACY_SCORE"] = base_accuracy
        base_data.loc[base_index, "FIT_TIME"] = base_fit_time
        base_data.loc[base_index, "SCORE_TIME"] = base_score_time
        
        base_dict[name] = classifier
        
        base_index+=1
        
    base_data = base_data.set_index("NAME")
    base_data = base_data.sort_values(by = "ROC_AUC_SCORE",ascending = False)
    
    return base_data, base_dict


# In[19]:


classifiers = [ 
                 #("RF", RandomForestClassifier(), rf_params),
                 ("CART", DecisionTreeClassifier() , cart_params),    
                 ("LR" , LogisticRegression() , lr_params),
                 #("GBM" , GradientBoostingClassifier() , gbm_params),
                 #("XGBoost" , XGBClassifier(), xgboost_params),
                 ("LightGBM", LGBMClassifier() , lightgbm_params),
    
]


# In[20]:


basic_data , basic_dict = base_advanced_model(classifiers, X_Scaled, y, cv = 2)


# In[21]:


basic_data


# In[22]:


basic_dict


# In[23]:


def advanced_hyperoptimization_model(classifiers, X, y, cv = 3):
    
    advanced_base = pd.DataFrame()
    advanced_index = 0
    advanced_dict = {}
    
    
    for name, classifier, params in classifiers:
        
        cv_results = cross_validate(estimator=classifier,
                                    X = X,
                                    y = y,
                                    cv = cv,
                                    n_jobs= -1, 
                                    verbose=0,
                                    scoring = ["roc_auc","accuracy"])
        
        base_roc_auc = cv_results["test_roc_auc"].mean()
        base_accuracy = cv_results["test_accuracy"].mean()
        
        base_roc_auc = round(base_roc_auc, 4)
        base_accuracy = round(base_accuracy, 4)
        
        print(f" {type(classifier).__name__.upper()} ".center(50,"#"),end = "\n\n")
        
        print("Before GridSearchCV Scores",end ="\n\n")
        print("Roc Auc Score : {}\nAccuracy Score : {}".format(base_roc_auc,base_accuracy))
        
        best_grid = GridSearchCV(estimator = classifier,
                                 param_grid= params,
                                 n_jobs=-1,
                                 verbose=0,
                                 cv=cv,
                                 scoring="roc_auc").fit(X,y)
        
        print(f"Best Params : {best_grid.best_params_}",end = "\n\n")
        
        final_classifier = classifier.set_params(**best_grid.best_params_)
        
        final_cv_results = cross_validate(estimator=final_classifier,
                                        X = X,
                                        y = y,
                                        cv = cv,
                                        n_jobs= -1, 
                                        verbose=0,
                                        scoring = ["roc_auc","accuracy"])
        
        advanced_roc_auc  = final_cv_results["test_roc_auc"].mean()
        advanced_accuracy = final_cv_results["test_accuracy"].mean()
        
        advanced_accuracy = round(advanced_accuracy,4)
        advanced_roc_auc = round(advanced_roc_auc,4)
        
        advanced_fit_time = round(final_cv_results["fit_time"].mean(), 4)
        
        print("After GridSearchCV Scores",end ="\n\n")
        print("Roc Auc Score : {}\nAccuracy Score : {}".format(advanced_roc_auc,advanced_accuracy)
              ,end ="\n\n")
        
        
        advanced_base.loc[advanced_index, "NAME"] = name
        advanced_base.loc[advanced_index, "BEFORE_ROC_AUC_SCORE"] = base_roc_auc
        advanced_base.loc[advanced_index, "BEFORE_ACCURACY_SCORE"] = base_accuracy
        advanced_base.loc[advanced_index, "AFTER_ROC_AUC_SCORE"] = advanced_roc_auc
        advanced_base.loc[advanced_index, "AFTER_ACCURACY_SCORE"] = advanced_accuracy
        advanced_base.loc[advanced_index, "AFTER_FIT_TIME"] = advanced_fit_time
            
        advanced_dict[name] = final_classifier
        
        advanced_index+=1
    
    
    advanced_base = advanced_base.set_index("NAME")
    advanced_base = advanced_base.sort_values(by = "AFTER_ROC_AUC_SCORE", ascending = False)
    
    return advanced_base, advanced_dict


# In[24]:


cart_params = {
                'max_depth': np.arange(5,20,5,dtype = int),
                "min_samples_split": np.arange(5,20,5,dtype = int)}

lr_params = {
             "C" : [1.0, 0.1, 0.01],
             "max_iter" : np.linspace(100,300,3,dtype = int)
             }


lightgbm_params = {
                    "learning_rate" : [0.1, 0.2],
                    #"max_depth" : [5,8,12],
                    "n_estimators" : [100,200,300],
                    "colsample_bytree" : [0.5, 1]
                  }

classifiers = [ 
                 #("RF", RandomForestClassifier(), rf_params),
                 ("CART", DecisionTreeClassifier() , cart_params),    
                 ("LR" , LogisticRegression() , lr_params),
                 #("GBM" , GradientBoostingClassifier() , gbm_params),
                 #("XGBoost" , XGBClassifier(), xgboost_params),
                 ("LightGBM", LGBMClassifier() , lightgbm_params),
    
]


# In[25]:


advanced_data, advanced_dict = advanced_hyperoptimization_model(classifiers,X_Scaled, y, cv = 2)


# In[26]:


advanced_data


# In[27]:


advanced_dict


# In[28]:


import pickle
path_model = "/Users/gokhanersoz/Desktop/GitHub/Fraud/Model"

for name in advanced_dict:
    advanced_dict[name].fit(X_Scaled, y)
    pd.to_pickle(advanced_dict[name], open(f"{path_model}/{name.upper()}.pkl","wb"))


# In[29]:


Lightgbm_proba = advanced_dict["LightGBM"].predict_proba(X_Scaled)[:,1]
LR_proba = advanced_dict["LR"].predict_proba(X_Scaled)[:,1]
Cart_proba = advanced_dict["CART"].predict_proba(X_Scaled)[:,1]


# In[30]:


lgbm_roc = roc_auc_score(y, Lightgbm_proba)
lr_roc = roc_auc_score(y, LR_proba)
cart_roc = roc_auc_score(y, Cart_proba)

lgbm_accuracy = accuracy_score(y , advanced_dict["LightGBM"].predict(X_Scaled))
lr_accuracy = accuracy_score(y , advanced_dict["LR"].predict(X_Scaled))
cart_accuracy = accuracy_score(y , advanced_dict["CART"].predict(X_Scaled))


lgbm_fpr, lgbm_tpr, lgbm_threshold = roc_curve(y, Lightgbm_proba)
lr_fpr, lr_tpr, lr_threshold = roc_curve(y, LR_proba)
cart_fpr, cart_tpr, cart_threshold = roc_curve(y, Cart_proba)

print("Train LightGBM Roc Auc Score : {}".format(round(lgbm_roc,4)))
print("Train LightGBM Accuracy Score : {}".format(round(lgbm_accuracy,4)),end = "\n\n")

print("Train LR Roc Auc Score : {}".format(round(lr_roc,4)))
print("Train LR Accuracy Score : {}".format(round(lr_accuracy,4)),end = "\n\n")

print("Train CART Roc Auc Score : {}".format(round(cart_roc,4)))
print("Train CART Accuracy Score : {}".format(round(cart_accuracy,4)),end = "\n\n")


# ##  VotingClassifiers

# In[31]:


from sklearn.ensemble import VotingClassifier


# In[32]:


models = [ (col,advanced_dict[col]) for col in advanced_dict] 

voting_classifier = VotingClassifier(estimators= models, voting = "soft", n_jobs=-1).fit(X_Scaled,y)

voting_proba = voting_classifier.predict_proba(X_Scaled)[:,1]
voting_roc = roc_auc_score(y, voting_proba)

voting_fpr, voting_tpr, voting_thresholds = roc_curve(y, voting_proba)


# In[33]:


def voting_classifiers(estimators,X,y,cv = 2):
    
    voting_data = pd.DataFrame()
    voting_index = 0
    voting_dict = {}
    
    voting_classifier = VotingClassifier(estimators=estimators,
                                         voting="soft",
                                         n_jobs=-1).fit(X,y)
    
    cv_results = cross_validate(estimator = voting_classifier,
                                    X = X,
                                    y = y,
                                    cv = cv,
                                    n_jobs=-1, 
                                    verbose = 0 ,
                                    scoring = ["roc_auc","accuracy"])
    
    
    voting_roc_auc = cv_results["test_roc_auc"].mean()
    voting_accuracy = cv_results["test_accuracy"].mean()
    voting_fit_time = cv_results["fit_time"].mean()
    voting_score_time = cv_results["score_time"].mean()
        
    voting_data.loc[voting_index, "NAME"] = type(voting_classifier).__name__
    voting_data.loc[voting_index, "ROC_AUC_SCORE"] = round(voting_roc_auc, 4)
    voting_data.loc[voting_index, "ACCURACY_SCORE"] = round(voting_accuracy, 4)
    voting_data.loc[voting_index, "FIT_TIME"] = round(voting_fit_time, 4)
    voting_data.loc[voting_index, "SCORE_TIME"] = round(voting_score_time, 4)
        
    voting_dict[name] = voting_classifier
        
    voting_data = voting_data.set_index("NAME").sort_values("ROC_AUC_SCORE")
    
    return voting_data, voting_dict


# In[34]:


voting_base , voting_dict = voting_classifiers(models, X_Scaled, y, cv = 2)


# In[35]:


voting_base


# In[36]:


pd.concat([basic_data,voting_base],axis = 0)


# In[37]:


print("Train LightGBM Roc Auc Score : {}".format(round(lgbm_roc,4)),end = "\n\n")
print("Train LR Roc Auc Score : {}".format(round(lr_roc,4)),end = "\n\n")
print("Train CART Roc Auc Score : {}".format(round(cart_roc,4)),end = "\n\n")
print("Train Voting Roc Auc Score : {}".format(round(voting_roc,4)),end = "\n\n")


# In[38]:


plt.figure(figsize = (15,10))

plt.plot(lgbm_fpr, lgbm_tpr, "r--" ,          label = f"{type(advanced_dict['LightGBM']).__name__} Curve (Area : {round(lgbm_roc,4)})" )

plt.plot(lr_fpr, lr_tpr, "b--", 
         label = f"{type(advanced_dict['LR']).__name__} Curve (Area : {round(lr_roc,4)})" )

plt.plot(cart_fpr, cart_tpr, "g--", 
         label = f"{type(advanced_dict['CART']).__name__} Curve (Area : {round(cart_roc,4)})" )

plt.plot(voting_fpr, voting_tpr, "y--", 
         label = f"VotingClassifiers Curve (Area : {round(voting_roc,4)})" )

plt.plot([0,1],[0,1],"black")


size = 15
plt.legend( loc  = "best",fontsize = size)
plt.xlabel("False Positive Rate" ,fontsize = size)
plt.ylabel("True Positive Rate", fontsize = size)
plt.title("ROC CURVE", fontsize = size)

plt.show()


# In[39]:


for name in advanced_dict:
    
    size=20
    cm = confusion_matrix(y, advanced_dict[name].predict(X_Scaled))
    
    name = type(advanced_dict[name]).__name__.upper()
    
    plt.figure(figsize = (10,5))
    sns.heatmap(data = cm ,
                annot=True,
                annot_kws={"size" : 20,"color" : "black"},
                fmt=".1f",
                cmap = "rainbow",
                cbar =False,
                xticklabels=["NoFraud","Fraud"],
                yticklabels=["NoFraud","Fraud"],
                linewidths=0.5,
                center = 0);
    
    
    plt.title(name, fontsize = size)
    plt.xlabel("Predicted Label", fontsize = size)
    plt.ylabel("True Label", fontsize = size)
    plt.show()


size=20
cm = confusion_matrix(y, voting_classifier.predict(X_Scaled))
    
name = type(voting_classifier).__name__.upper()
    
plt.figure(figsize = (10,5))
sns.heatmap(data = cm ,
            annot=True,
            annot_kws={"size" : 20,"color" : "black"},
            fmt=".1f",
            cmap = "rainbow",
            cbar =False,
            xticklabels=["NoFraud","Fraud"],
            yticklabels=["NoFraud","Fraud"],
            linewidths=0.5,
            center = 0);
    
    
plt.title(name, fontsize = size)
plt.xlabel("Predicted Label", fontsize = size)
plt.ylabel("True Label", fontsize = size)
plt.show()


# In[40]:


def feature_importance(model, X , nums = 20, save = False):
    
    imp_feature = pd.DataFrame()
    imp_feature["Values"] = model.feature_importances_
    imp_feature["Name"] = [col for col in X.columns]
    imp_feature = imp_feature.sort_values(by = "Values", ascending = False)    
        
    size = 15
    plt.figure(figsize = (15,10))
    sns.barplot(x = "Values", y ="Name", data = imp_feature[:nums]
                ,palette = "viridis")
    plt.xlabel("Values",fontsize = size)
    plt.ylabel("NAME", fontsize = size)
    plt.title(f"For {type(model).__name__.upper()} Feature Importance",
             fontsize = size)
    plt.show()
    
    if save:
        
        plt.savefig(type(model).__name__ + "_importance.png")


# In[41]:


feature_importance(advanced_dict["LightGBM"], X , nums = 30)


# In[42]:


feature_importance(advanced_dict["CART"], X , nums = 30, save = True)


# In[43]:


def feature_importance_coef(model, X , nums = 20, save = False):
    
    imp_feature = pd.DataFrame()
    imp_feature["Values"] = model.coef_[0]
    imp_feature["Name"] = [col for col in X.columns]
    imp_feature = imp_feature.sort_values(by = "Values", ascending = False)    
        
    size = 15
    plt.figure(figsize = (15,10))
    sns.barplot(x = "Values", y ="Name", data = imp_feature[:nums]
                ,palette = "viridis")
    plt.xlabel("Values",fontsize = size)
    plt.ylabel("NAME", fontsize = size)
    plt.title(f"For {type(model).__name__.upper()} Feature Importance",
             fontsize = size)
    plt.show()
    
    if save:
        
        plt.savefig(type(model).__name__ + "_importance.png")


# In[44]:


feature_importance_coef(advanced_dict["LR"], X, nums = 20)


# # Analyzing Model Complexity with Learning Curves

# In[45]:


from sklearn.model_selection import validation_curve


# In[46]:


lightgbm_params


# In[47]:


def Validation_Curve(model,X,y,param_name,param_range,cv = 2, scoring = "roc_auc"):
    
    train_scores, test_scores = validation_curve(estimator=model,
                                                 X = X,
                                                 y = y,
                                                 cv = cv,
                                                 n_jobs=-1,
                                                 verbose=0,
                                                 param_name = param_name,
                                                 param_range = param_range,
                                                 scoring = scoring)
    
    
    train_mean_scores = np.mean(train_scores,axis = 1)
    test_mean_scores = np.mean(test_scores, axis = 1)
    
    
    plt.figure(figsize = (15,8))
    size = 15
    
    plt.plot(param_range, train_mean_scores,
             label = "Training Scores")
    
    plt.plot(param_range, test_mean_scores,
             label = "Validation Scores")
    
    plt.legend(loc = "best" , fontsize = size)
    plt.xlabel(f"Param Range : {param_range}\nParam Name : {param_name}",
               fontsize = size)
    plt.ylabel(f"{scoring}", fontsize = size)
    plt.title(f"{type(model).__name__.upper()} İçin Validation Curve", 
              fontsize = size)
    
    plt.show()


# In[48]:


for cols in classifiers:
    
    name = cols[0]
    classifier = cols[1]
    params = cols[2]
    
    model = advanced_dict[name]
    
    # params dict içersinde fakat isimleri dönüyor ...
    
    for col in params:
        
        Validation_Curve(model = model,
                         X=X_Scaled,
                         y = y,
                         param_name=col,
                         param_range=params[col])        


# In[49]:


# Burda Değerler birbiri ile eş zamanlı gerçekleşmiyor!!!
# Tek Tek olarak el alınıyor bunu göz önüne alalım 
# Hem GridSearchCV kullanarak ne kadar doğru bir seçim yapıldığına bakıyoruz
# Değerler Bizi Doğru olanları seçildiğini gösteriyor...

for cols in classifiers:
    
    name = cols[0]
    classifier = cols[1]
    params = cols[2]    
    
    print(f"{type(classifier).__name__.upper()} Params Değerleri\n")
     
    for col in params:
        
        print("{} : {}".format(col,advanced_dict[name].get_params()[col]))
        
    print("\n\n")


# ## KFOLD - STRATIFIEDKFOLD

# In[50]:


def kfold_models(model , X, y, n_splits = 5, kfold = True):
    """
    
    StratifiedKFold için kfold = False Yapın !!!
    
    """
    kfold_data = pd.DataFrame()
    kfold_index = 0
    kfold_list = []
    
    
    if kfold:
        
        print("KFold Using ....", end = "\n\n")
        print(f"KFold Splits = {n_splits}",end = "\n\n")
        fold_name = "KFold_"
        KFOLD = KFold(n_splits= n_splits , shuffle=True)
    
    else:
        
        print("StratifiedKFold Using .....", end= "\n\n")
        print(f"StratifiedKFold Splits = {n_splits}", end = "\n\n")
        fold_name = "StratifiedKFold_"
        KFOLD = StratifiedKFold(n_splits= n_splits ,shuffle=True)
        
        
    for n_fold, (train_idx,test_idx) in enumerate(KFOLD.split(X,y)):
        
        
        X_train,X_test = X.iloc[train_idx],X.iloc[test_idx] 
        y_train,y_test = y.iloc[train_idx],y.iloc[test_idx]
        
        model_fit = model.fit(X_train, y_train)
        
        
        y_pred = model_fit.predict(X_test)
        y_proba =  model_fit.predict_proba(X_test)[:,1]
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr,tpr,thresholds = roc_curve(y_test, y_proba)
        
        kfold_list.append( (n_fold,fpr,tpr,roc_auc) )
        
        kfold_data.loc[ kfold_index, fold_name +"NFOLD"] = n_fold + 1
        kfold_data.loc[ kfold_index, fold_name +"ROC_AUC_SCORE"] = round(roc_auc, 4)
        kfold_data.loc[ kfold_index, fold_name +"ACCURACY_SCORE"] = round(accuracy,4)
        
        kfold_index+=1
        
    kfold_data = kfold_data.set_index(fold_name + "NFOLD")
        
    print("Finish !!!! ")
        
        
    return kfold_list, kfold_data


# In[51]:


def kfold_roc_auc_graph(roc_auc_values):
    
    mean_roc = []
    plt.figure(figsize = (15,8))
    
    for nfold,fpr,tpr,roc_scores in roc_auc_values:
        
        mean_roc.append(roc_scores)
        scores = round(roc_scores,4)
        plt.plot(fpr,tpr,
                 label = f"For K-Folds : {nfold+1}, ROC AUC (Area : {scores})")
    
    mean_roc_cal = np.mean(mean_roc)
    mean_roc_cal = round(mean_roc_cal, 4)
    plt.plot([0,1],[0,1],"black")
    plt.legend(loc = "best")
    plt.xlabel("False Positive Rate",fontsize = size)
    plt.ylabel("True Positive Rate",fontsize = size)
    plt.title(f"ROC AUC CURVE (AVERAGE AREA : {mean_roc_cal}))",fontsize = size)
    plt.show()    


# In[52]:


kfold_list, kfold_data = kfold_models(model = advanced_dict["LightGBM"], 
                                      X = X_Scaled, 
                                      y = y, 
                                      n_splits=2 , 
                                      kfold = True)


# In[53]:


Stratified_kfold_list, Stratified_kfold_data = kfold_models(model = advanced_dict["LightGBM"], 
                                                            X = X_Scaled, 
                                                            y = y, 
                                                            n_splits=2 , 
                                                            kfold = False)


# In[54]:


kfold_data


# In[55]:


Stratified_kfold_data


# In[56]:


kfold_roc_auc_graph(kfold_list)


# In[57]:


kfold_roc_auc_graph(Stratified_kfold_list)


# ##  Let's create our Forecast Data

# In[58]:


submission_df = pd.DataFrame()
submission_df["TransactionID"] = test["TransactionID"]


# In[59]:


test.head()


# In[60]:


print("Test Shape : {}".format(test.shape))


# In[61]:


test_cat_cols = ['ProductCD',
 'card4',
 'card6',
 'P_emaildomain',
 'R_emaildomain',
 'M1',
 'M2',
 'M3',
 'M4',
 'M5',
 'M6',
 'M7',
 'M8',
 'M9',
 'id-12',
 'id-15',
 'id-16',
 'id-23',
 'id-27',
 'id-28',
 'id-29',
 'id-30',
 'id-31',
 'id-33',
 'id-34',
 'id-35',
 'id-36',
 'id-37',
 'id-38',
 'DeviceType',
 'DeviceInfo',
 'day_risk',
 'hours_risk',
 'Risk_TransactionAmt',
 'Risk_ProductCD',
 'card3_feature',
 'addr1_risk',
 'Dist1_Risk',
 'Dist2_Risk',
 'OS_id_30',
 'Version_id_30',
 'Browser_id_31',
 'Version_id_31',
 'Width_id_33',
 'Height_id_33',
 'DeviceCorp']


# In[62]:


Test_Scaled = sc.transform(test[train_num_cols])
Test_Scaled = pd.DataFrame(data = Test_Scaled, columns = train_num_cols)

last_test = pd.concat([Test_Scaled,test[test_cat_cols]], axis = 1)

print("Test Shape : {}".format(last_test.shape))


# In[63]:


submission_df["isFraud"] = advanced_dict["LightGBM"].predict_proba(last_test)[:,1]
submission_df.head()


# In[64]:


submission_df["isFraud"].value_counts()


# In[65]:


submission_df.to_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/submission_df.csv",index=False)


# In[66]:


pd.read_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/submission_df.csv")


# In[ ]:




