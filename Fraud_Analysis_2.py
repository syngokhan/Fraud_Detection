#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.set_option("display.max_columns",None)
pd.set_option("display.float_format" , lambda x : "%.4f" %x)
pd.set_option("display.width", 200)


# In[3]:


from warnings import filterwarnings
import gc
filterwarnings("ignore", category= DeprecationWarning)
filterwarnings("ignore", category= FutureWarning)
filterwarnings("ignore")


# In[4]:


get_ipython().run_cell_magic('time', '', '\npath = "/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data"\n\ntrain = pd.read_csv(f"{path}/train.csv",index_col=0)\ntest = pd.read_csv(f"{path}/test.csv",index_col=0)\n\nprint("Train Shape : {}".format(train.shape))\nprint("Test Shape : {}".format(test.shape))\ngc.collect()')


# In[5]:


train_cat_cols = [col for col in train.columns if train[col].dtype == "object"]
test_cat_cols = [col for col in test.columns if test[col].dtype == "object"]

for col in train_cat_cols:
    
    train[col].fillna("UnKnown",inplace = True)
    
for col in test_cat_cols:
    
    test[col].fillna("UnKnown",inplace = True)


# In[6]:


pd.Series(train_cat_cols).to_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data/train_cats.csv")
pd.Series(test_cat_cols).to_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data/test_cats.csv")


# In[7]:


# Control !!!

for col in train_cat_cols:
    if train[col].isnull().sum() > 0:
        print(col)
        
for col in test_cat_cols:
    if test[col].isnull().sum() > 0:
        print(col)


# In[8]:


from sklearn.decomposition import PCA


# In[9]:


data_V = ["V"+str(i) for i in range(1,340) if i != 107]

for V in data_V:
    train[V].fillna(0, inplace = True)
    test[V].fillna( 0, inplace = True)


# In[10]:


train[data_V].head()


# In[11]:


test[data_V].head()


# In[12]:


from sklearn.decomposition import PCA

pca = PCA()

pca_model = pca.fit_transform(train[data_V])


# In[13]:


plt.figure(figsize = (15,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_)* 100,"go-",markersize = 10)

size = 15
plt.title("V1-V339 İçin PCA Boyut İndirgeme\nSeçilen Nokta Kırmızı ile Berlirlendi",fontsize = size)
plt.xlabel("V Miktarı",fontsize = size)
plt.ylabel("Açıklanabilirlik Oranı",fontsize = size)
plt.xticks(range(0,340,10),rotation = 45)
plt.xlim([-1,15])

plt.axhline(99.550284,ls = "--",color = "r")
plt.axvline(2,ls = "--",c = "r")
plt.show()


# In[14]:


# Final Model For PCA

final_pca = PCA(n_components=3,)
final_pca_model_train = final_pca.fit_transform(train[data_V])


# In[15]:


final_pca_model_test = final_pca.transform(test[data_V])


# In[16]:


V_Dataframe_Train = pd.DataFrame(data = final_pca_model_train, columns = ["PCA_V_"+str(i) for i in range(1,4)])
V_Dataframe_Test = pd.DataFrame(data = final_pca_model_test, columns = ["PCA_V_"+str(i) for i in range(1,4)])


# In[17]:


print("PCA V Train : {}".format(V_Dataframe_Train.shape))
print("PCA V Test  : {}".format(V_Dataframe_Test.shape))
print("\n\n")
print("Train Shape : {}".format(train.shape))
print("Test Shape : {}".format(test.shape))
gc.collect()


# In[18]:


V_Dataframe_Train.isnull().sum()


# In[19]:


V_Dataframe_Test.isnull().sum()


# In[20]:


train.drop(data_V,axis = 1, inplace = True)
test.drop(data_V, axis = 1, inplace = True)

print("Train Shape : {}".format(train.shape))
print("Test Shape : {}".format(test.shape))
gc.collect()


# In[21]:


train.reset_index().index.tolist() == V_Dataframe_Train.index.tolist()


# In[22]:


#train.reset_index(drop=True)


# In[23]:


train = pd.concat([train.reset_index(drop=True) ,V_Dataframe_Train]   ,axis = 1 )
test  = pd.concat([test.reset_index(drop=True)  ,V_Dataframe_Test ]   ,axis = 1 )

print("Train Shape : {}".format(train.shape))
print("Test Shape : {}".format(test.shape))


# In[24]:


gc.collect()


# In[25]:


train.head()


# In[26]:


test.head()


# In[27]:


train_num_cols = [col for col in train.columns if train[col].dtype != "object" and col not in ["TransactionID",
                                                                                         "isFraud"]]
test_num_cols = [col for col in test.columns if test[col].dtype != "object" and col not in ["TransactionID",
                                                                                         "isFraud"]]

for col in train_num_cols:
    
    print("{} NaN Values: {}".format(col, train[col].isnull().sum()))


# In[28]:


pd.Series(train_num_cols).to_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data/train_nums.csv")
pd.Series(test_num_cols).to_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data/test_nums.csv")


# In[29]:


def replace_inf_to_nan(dataframe,col):
    
    dataframe[col].replace([np.inf,-np.inf], np.nan,inplace = True)


# In[30]:


def check_inf(dataframe,col):
    
    sum_inf = sum(np.isinf(dataframe[col]))
    sum_neginf = sum(np.isneginf(dataframe[col]))
    if sum_inf > 0 or sum_neginf > 0:
        
        print("{} Columns Have INF or NegINF Values ...".format(col))


# In[31]:


for col in train_num_cols:
    
    check_inf(train,col)


# In[32]:


for col in test_num_cols:
    
    check_inf(test,col)


# In[33]:


for col in train_num_cols:
    
    replace_inf_to_nan(train,col)
    
for col in test_num_cols:
    
    replace_inf_to_nan(test,col)


# In[34]:


for col in train_num_cols:
    
    check_inf(train,col)
    
for col in test_num_cols:
    
    check_inf(test,col)


# In[35]:


#for col in test_cat_cols:
    
#    print("{} NaN Values : {}".format(col, test[col].isnull().sum()))

#for col in train_cat_cols:
    
#    print("{} NaN Values : {}".format(col, train[col].isnull().sum()))


# In[36]:


def missing_graph(dataframe,cols,percentage = 50):
    
    
    data = pd.DataFrame()
    data["Missing(%)"] = dataframe[cols].isnull().sum() / len(dataframe) * 100
    data["Name"] = [col for col in cols]
    data = data.sort_values("Missing(%)",ascending = False)

    plt.figure(figsize = (15,8))
    
    try:
        sns.barplot(x = "Missing(%)", y = "Name", data = data[data["Missing(%)"] > percentage])
    except:
        pass
    
    size = 15
    
    plt.ylabel("Name",fontsize = size)
    plt.xlabel("Missing Percentage Values", fontsize = size)
    plt.title(f"Eksik Değerlerin Yüzdelik Değerleri\n(%{percentage} Eksik Değer)", fontsize = size)
    plt.show()
    
    return data


# In[37]:


def na_control(dataframe):
    
    liste = []
    
    for col in dataframe.columns:

        na_values = dataframe[col].isnull().sum()

        if na_values > 0:

            liste.append( (col,na_values) )
    
    if len(liste) > 0:
        
        for col_name,na_values in liste:
            
            print(f"{col_name.upper()} Na Values : {na_values}")
    
    else:
        
        print("Boş Değer Yok !!!!!")


# In[38]:


missing_nums = missing_graph(train,train_num_cols,percentage=10)


# In[39]:


missing_cats = missing_graph(train,train_cat_cols,percentage=10)


# In[40]:


missing_cats.T


# In[41]:


missing_nums.T


# In[42]:


train_missing_gt_50 = missing_nums[missing_nums["Missing(%)"] > 50]["Name"].tolist()

pd.DataFrame(train_missing_gt_50).T


# In[43]:


test_missing_gt_50 = pd.Series(train_missing_gt_50).replace("_","-",regex = True).tolist()

pd.DataFrame(test_missing_gt_50).T


# In[44]:


for data in [train, test]:
    na_control(data)


# In[45]:


for col in train_num_cols:
    
    col_median = train[col].median()
    
    train[col].fillna(col_median, inplace = True)    


# In[46]:


for col in test_num_cols:
    
    col_median = test[col].median()
    
    test[col].fillna(col_median, inplace = True)        


# In[47]:


for data in [train, test]:
    na_control(data)


# In[48]:


len(train_cat_cols) == len([col for col in train.columns if train[col].dtype == "object"])


# In[49]:


len(test_cat_cols) == len([col for col in test.columns if test[col].dtype == "object"])


# In[50]:


len(train_cat_cols),len([col for col in train.columns if train[col].dtype == "object"])


# In[51]:


len(test_cat_cols) ,len([col for col in test.columns if test[col].dtype == "object"])


# In[52]:


def label_encoder(dataframe,cat_cols):
    
    from sklearn.preprocessing import LabelEncoder
    
    for cat in cat_cols:
        
        dataframe[cat] = LabelEncoder().fit_transform(dataframe[cat])


# In[53]:


label_encoder(train, train_cat_cols)
label_encoder(test , test_cat_cols)


# In[54]:


def check_object(dataframe):
    
    liste = [col for col in dataframe.columns if dataframe[col].dtype == "object"]
    
    if len(liste) > 0 :
        
        for col in liste:
            
            print(f"{col.upper()} : Bu Feature Object Değerdir....")
            
    else:
        
        print("Object Değer Yoktur !!!!")


# In[55]:


check_object(train)
check_object(test)


# In[56]:


train[train_cat_cols].head()


# In[57]:


test[test_cat_cols].head()


# In[58]:


train.head()


# In[59]:


test.head()


# In[60]:


na_control(train)
na_control(test)


# In[61]:


# ALL DATA (NO DROP)

train.to_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data/all_train.csv")
test.to_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data/all_test.csv")


# In[62]:


# DROP DATA

train.drop(train_missing_gt_50, axis = 1).to_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data/drop_train.csv")
test.drop(test_missing_gt_50, axis = 1).to_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data/drop_test.csv")

