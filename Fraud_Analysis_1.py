#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.set_option("display.max_columns" , None)
pd.set_option("display.float_format" , lambda x : "%.4f" % x)


# In[3]:


from warnings import filterwarnings
import gc

filterwarnings("ignore" , category = DeprecationWarning)
filterwarnings("ignore" , category = FutureWarning)
filterwarnings("ignore")


# In[4]:


path = "/Users/gokhanersoz/Desktop/GitHub/Fraud/ieee-fraud-detection"


# In[5]:


get_ipython().run_cell_magic('time', '', '\n# Transaction : İşlem\n# İdentity : Kimlik\n\ntrain_transaction = pd.read_csv(f"{path}/train_transaction.csv")\ntest_transaction = pd.read_csv(f"{path}/test_transaction.csv")\n\ntrain_identity = pd.read_csv(f"{path}/train_identity.csv")\ntest_identity = pd.read_csv(f"{path}/test_identity.csv")')


# In[6]:


print("Train Transcation Shape : {}".format(train_transaction.shape))
print("Train Identity Shape : {}".format(train_identity.shape))
print("\n")
print("Test Transcation Shape : {}".format(test_transaction.shape))
print("Test Identity Shape : {}".format(test_identity.shape))


# In[7]:


get_ipython().run_cell_magic('time', '', '\ntrain = pd.merge(left = train_transaction,\n                 right = train_identity,\n                 on = "TransactionID", \n                 how = "left")\n\ntest = pd.merge(left = test_transaction,\n                right = test_identity,\n                on = "TransactionID",\n                how = "left")\n\ndel train_transaction,train_identity, test_transaction, test_identity\ngc.collect()')


# In[8]:


print("Train Shape : {}".format(train.shape))
print("Test Shape : {}".format(test.shape))


# In[9]:


train.tail(3)


# In[10]:


test.head(3)


# ##  EDA
# 
# **reference** : https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203
# 
# **
# 
# ``TranscationDT``:
# 
# Belirli bir referans tarih saatinden Timedelta (Gerçek bir zaman damgası değil !!!!)
# 
# “ TransactionDT ilk değeri 86400 yani bir gün içindeki saniye sayısına (60 * 60 * 24=86400) tekabül ediyor yani birimi saniye sanıyorum. Bunu kullanarak, maksimum değerin 183. güne karşılık gelen 15811131 olduğu için verilerin 6 ayı kapsadığını biliyoruz. ”
# 
# ``TranscationAmt``:
# 
# USD cinsinden işlem ödeme tutarı
# 
# “ Bazı işlem tutarlarında, ondalık noktanın sağında üç ondalık basamak bulunur. Üç ondalık basamağa ve boş bir addr1 ve addr2 alanına bir bağlantı var gibi görünüyor. Bunların yabancı işlemler olması ve örneğin 12. satırdaki 75.887'nin bir döviz tutarının bir döviz kuru ile çarpılmasının sonucu olması mümkün müdür? ”
# 
# ``dist``:
# 
# Mesafe
# 
# " Fatura Adresi, Posta Adresi, Posta Kodu, Ip Adresi, Telefon Alanı Vb. Arasındaki (Sınırlı Olmayan) Mesafeler. "
# 
# 
# ``C1-C14``:
# 
# Ödeme Kartıyla Kaç Adresin Ilişkilendirildiği Gibi Sayma Vb. Gerçek Anlam Maskelenir.
# 
# “ C1-15 değişkenlerinde daha fazla sayı örneği verebilir misiniz? Bunlar, kullanıcıyla ilişkili telefon numaraları, e-posta adresleri, adlar gibi mi? 15'i düşünemiyorum. ”
# 
# " Tahmininiz iyi, ayrıca cihaz, ipaddr, billingaddr vb. gibi. Bunlar hem alıcı hem de alıcı içindir, bu da sayıyı ikiye katlar. ”
# 
# ``D1-D15``:
# 
# Önceki Işlem Arasındaki Günler Vb. Gibi Zaman Deltası
# 
# ``V1-V339``:
# 
# Vesta, sıralama, sayma ve diğer varlık ilişkileri dahil olmak üzere zengin özellikler tasarladı.
# 
# "Örneğin, bir IP ve e-posta veya adresle ilişkili ödeme kartının 24 saatlik zaman aralığında kaç kez göründüğü vb."
# 
# "Tüm Vesta özellikleri sayısal olarak türetilmiştir. Bazıları bir kümeleme, bir zaman aralığı veya koşul içindeki siparişlerin sayısıdır, bu nedenle değer sonludur ve sıralamaya (veya sıralamaya) sahiptir. Hiçbirini ele almanızı tavsiye etmem. kategorik olarak. Bunlardan herhangi biri şans eseri ikili ile sonuçlandıysa, denemeye değer olabilir."
# 
# 
# **
# 
# ``ProductCD``: 
# 
# Ürün Kodu, Her Işlem Için Ürün
# 
# “Ürünün gerçek bir 'ürün' olması gerekmez (alışveriş sepetine eklenecek bir ürün gibi). Her türlü hizmet olabilir.”
# 
# ``card1 - card6``: 
# 
# Kart Türü, Kart Kategorisi, Veren Banka, Ülke Vb. Gibi Ödeme Kartı Bilgileri.
# 
# ``M1-M9``: 
# 
# Karttaki Isimler Ve Adres Vb. Gibi Eşleşme
# 
# ``addr1-addr2``:
# 
# Adres: Adres
# Her Iki Adres De Alıcı Içindir;
# Faturalandırma Bölgesi Olarak Addr1;
# Fatura Ülkesi Olarak Addr2
# 
# ``P_emaildomain --- R_emaildomain``:
# 
# Müşteri ve Alıcı e-posta alanı.
# 
# “ Belirli Işlemlerin Alıcıya Ihtiyacı Yoktur, Bu Nedenle R_Emaildomain Boştur. ”
# 
# ``DeviceType ; DeviceInfo; id_01 - id_11; id_12 - id_38``:
# 
# Bu tablodaki değişkenler, işlemlerle ilişkili kimlik bilgileri – ağ bağlantı bilgileri (IP, ISP, Proxy, vb.) ve dijital imzadır (UA/tarayıcı/os/sürüm, vb.).
# Vesta'nın dolandırıcılık koruma sistemi ve dijital güvenlik ortakları tarafından toplanırlar.
# (Alan adları maskelenmiştir ve gizlilik koruması ve sözleşme sözleşmesi için ikili sözlük sağlanmayacaktır)
# 
# “id01 ila id11, cihaz derecelendirmesi, ip_domain derecelendirmesi, proxy derecelendirmesi vb. gibi Vesta ve güvenlik ortakları tarafından toplanan kimlik için sayısal özelliklerdir. Ayrıca, hesap oturum açma süreleri/giriş başarısızlığı süreleri, bir hesabın ne kadar süreceği gibi davranışsal parmak izlerini de kaydetti. sayfada kaldı, vb. Güvenlik ortağı T&C nedeniyle bunların tümü ayrıntılandırılamıyor. Umarım bu özelliklerin temel anlamını kavrarsınız ve sayısal/kategorik olarak bahsederek uygunsuz bir şekilde işlemezsiniz.”

# In[11]:


def summary(dataframe):
    
    data = pd.DataFrame()
    data["DTYPES"] = dataframe.dtypes
    data["NAME"] = [col for col in dataframe.columns]
    data["NUNIQUE"] = [dataframe[col].nunique() for col in dataframe.columns]
    data["Missing(%)"] = dataframe.isnull().sum() / len(dataframe) * 100
    
    data = data.reset_index()
    data.drop("index" , axis = 1, inplace= True)
    data = data.sort_values(by = "Missing(%)", ascending = False)
    
    return data


# In[12]:


get_ipython().run_cell_magic('time', '', '\ntrain_summary = summary(train)\ntest_summary = summary(test)')


# In[13]:


train_summary.head()


# In[14]:


test_summary.head()


# In[15]:


def missing_percentage(summary_func):
    
    """
    
    Summary function with using....
    
    """
    
    for i in range(0,100,10):
        
        percentage = summary_func[summary_func["Missing(%)"] > i]["Missing(%)"].count()                      / len(summary_func) * 100
        
        percentage = round(percentage, 4)
        
        print(f"Missing Değeri %{i}'Dan Büyük Olanların Değerine Karşılık Data %{percentage} Boştur..")
        


# In[16]:


# Train

missing_percentage(train_summary)


# In[17]:


# Test

missing_percentage(test_summary)


# In[18]:


one_train = [col for col in train.columns if train[col].nunique() == 1]
one_test = [col for col in test.columns if test[col].nunique() == 1]

print("Bir Eşsiz Değere Sahip Train İçin : {}".format(one_train))
print("Bir Eşsiz Değere Sahip Test İçin : {}".format(one_test))


# In[19]:


train.drop("V107", axis = 1, inplace = True)
test.drop("V107", axis = 1, inplace = True)


# ## Hedef Değişkeni
# 
# ### Hedef değişkeni aşırı dengesiz ve sadece %0.03 kısmı Fraud olarak tanımlanmıştır.

# In[20]:


isFraud_percentage = train.isFraud.value_counts() / len(train)
isFraud_Data = pd.DataFrame(isFraud_percentage)
isFraud_Data = isFraud_Data.reset_index()
isFraud_Data.columns = ["Transcation", "IsFraud(%)"]


size = 15
plt.figure(figsize = (15,8))
fraud = sns.barplot(x = "Transcation", y = "IsFraud(%)", data = isFraud_Data)

for index, row in enumerate(isFraud_Data["IsFraud(%)"]):
    
    fraud.text(x = index,y =  row , s = round(row,3), size = size , color = "black", ha = "center")

plt.xlabel("Transcation", fontsize = size)
plt.ylabel("IsFraud", fontsize = size)
plt.title("Is Fraud Distribution \nFraud : 1 | NoFraud : 0", fontsize = size)
plt.show()


# # TranscationDT - TimeDelta From a Given Reference DateTime

# In[20]:


print("Train TranscationDT Max : {}".format(train["TransactionDT"].max()))
print("Train TranscationDT Min : {}".format(train["TransactionDT"].min()))
print("\n")
print("Test TranscationDT Max : {}".format(test["TransactionDT"].max()))
print("Test TranscationDT Min : {}".format(test["TransactionDT"].min()))


# In[21]:


# (60*60*24) -- > 1 Gün
# (60*60) --> 1 Saat

train_span = (train["TransactionDT"].max() - train["TransactionDT"].min()) / (60*60*24)

test_span = (test["TransactionDT"].max() - test["TransactionDT"].min()) / (60*60*24)

total_span = (test["TransactionDT"].max() - train["TransactionDT"].min()) / (60*60*24)

gap_span = (test["TransactionDT"].min() - train["TransactionDT"].max()) / (60*60*24)


print(f"Toplam Veri Setinin Zaman Aralığı {total_span} gündür.\n\n"
      f"Toplam Train Setinin Zaman Aralığı {train_span} gündür.\n\n"
      f"Toplam Test Setinin Zaman Aralığı {test_span} gündür.\n\n"
      f"Toplam Train ile Test Arası Boşluk {gap_span} gündür.\n\n")


# In[23]:


size = 15

plt.figure(figsize = (15,8))
sns.distplot(train["TransactionDT"], kde = False , label = "Train",hist_kws={"edgecolor":"black"})
sns.distplot(test["TransactionDT"],  kde = False , label = "Test", hist_kws={"edgecolor":"black"} )
plt.title("TransactionDT Dağılımı Saniye Cinsinden" , fontsize = size)
plt.xlabel("TransactionDT" , fontsize = size)
plt.legend(loc = "upper right")
plt.show()


# ## Days

# In[24]:


# Bile bile yapıyor - 1 çünkü 0 dan günü başlatırıyorum....!!!!
# Sonra + 1 yapılıyor değişen bişey olmuyor !!

# 5 // 3 = 1 ,
# 5 %  3 = 2 ,
# 5 / 3 = 1.666


# In[22]:


for i in range(0,7):
    
    print(f"i : {i}  , {i // 4}")


# In[23]:


(((train["TransactionDT"] // (60*60*24) - 1) % 7) ) + 1


# In[24]:


train["days"] = (((train["TransactionDT"] // (60*60*24) -1 ) % 7) + 1) 
test["days"]  = (((test["TransactionDT"] // (60*602*24) -1 ) % 7) + 1)


# In[25]:


data_day = train.groupby("isFraud")["days"].value_counts(normalize = True).mul(100).rename("percentage").           reset_index().sort_values("days")



plt.figure(figsize = (15,8))
data_bar = sns.barplot(x = "days" , y = "percentage", data = data_day, hue = data_day["isFraud"],
                       palette="viridis")

for p in data_bar.patches:
    data_bar.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    
    
size = 15    
plt.xlabel("Day",fontsize = size)
plt.ylabel("Percentage", fontsize = size)
plt.title("Günlere Göre Fraud Gerçekleşme Oranları",fontsize = size)
plt.legend(loc = "upper center", fontsize = size)
plt.show()


# In[26]:


train.groupby("isFraud")["days"].value_counts(normalize = True).rename("percentage").mul(100).reset_index().sort_values("days").T


# In[27]:


train.groupby("days")["isFraud"].value_counts(normalize = True).rename("percentage").mul(100).reset_index().sort_values("days").T


# In[28]:


(train.groupby("days")["isFraud"].value_counts() / len(train["days"])).rename("percentage").mul(100).reset_index().sort_values("days").T


# In[29]:


(train.groupby("isFraud")["days"].value_counts() / len(train["days"])).rename("percentage").mul(100).reset_index().sort_values("days").T


# In[30]:


sum(train["days"] == 2), 2963 / (2963 + 76871)


# In[31]:


train[train["days"] == 2]["isFraud"].value_counts()


# In[32]:


# Hist plot günün kaçtane içerdiğini gösteriyor
# Biz bunun üzerine günlerin ortalama fraud koyarak hangi gün daha yüksek bir fraud gerçekleşmiş onu
# görüyoruz !!!

size = 15
plt.figure(figsize = (15,8))

without_hue_day_mean = train.groupby("days")["isFraud"].mean()

plt.plot(without_hue_day_mean, "r--",marker="o",markersize = size)
plt.ylabel("Fraud Ortalaması Günlere Göre",fontsize = size)

plt.twinx()
plt.hist(train["days"],bins = 7, alpha = 0.3, edgecolor = "black")

plt.xlabel("Days", fontsize = size)

plt.ylabel("İşlem Sayısı" , fontsize = size)
plt.title("Maviler İşlem sayısı | Kırmızılar Fraud Ortalaması" , fontsize = size)
plt.grid(True)
plt.tight_layout()
plt.show()


# ***Days Değişkeni için scorelar belirleyelim***
# 
# * 1,2,7 Yüksek Risk
# * 3, 6 Orta Risk
# * 4, 5 Düşük Risk

# In[33]:


def day_risk(dataframe):
    
    day = dataframe
    
    if day == 1 or day == 2 or day == 7:
        return "HighRiskDays"
    
    elif day == 3 or day == 6:
        return "MiddleRiskDays"
    
    else:
        return "LowerRiskDays"


# In[34]:


train["day_risk"] = train["days"].apply(day_risk)
test["day_risk"] = test["days"].apply(day_risk)


# In[35]:


train["day_risk"].unique(),test["day_risk"].unique()


# In[36]:


train[["days","day_risk"]].head(2000).T


# In[37]:


#train[train["day"] == 6][["day","day_risk"]]


# ## Hours

# In[38]:


train["hours"] = ((train["TransactionDT"] // (60*60)) % 24) + 1
test["hours"] = ((test["TransactionDT"] // (60*60) ) % 24) + 1


# In[39]:


train["hours"].nunique(),test["hours"].nunique()


# In[40]:


hours_data = train.groupby("isFraud")["hours"].value_counts(normalize = True).mul(100).rename("percentage").reset_index().sort_values("hours")

plt.figure(figsize = (15,8))

hours_bar = sns.barplot(x = "hours", y = "percentage", data = hours_data , hue = hours_data["isFraud"],
                        palette = "viridis")


for p in hours_bar.patches:
    hours_bar.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

size = 15
plt.title("Saatlere Göre Fraud Gerçekleşme Oranı" , fontsize = size)
plt.ylabel("Percentage", fontsize = size)
plt.xlabel("Hours", fontsize = size)
plt.legend(loc = "upper center",fontsize = size)
plt.show()


# In[41]:


train[train["days"] == 1]["isFraud"].value_counts()


# In[42]:


3550 / (94952 + 3550)


# In[43]:


without_hue_hours_mean = train.groupby("hours")["isFraud"].mean()

size = 15

plt.figure(figsize = (15,8))

plt.plot(without_hue_hours_mean, "r--", marker = "o", markersize = size,)
plt.ylabel("Fraud Ortalaması Saate Göre",fontsize = size)
plt.xlabel("Hours",fontsize = size)

plt.twinx()
plt.hist(train["hours"], bins = 24,edgecolor = "black", alpha = .3)

plt.title("İşlem Sayıları ve Fraud Ortalaması " ,fontsize = size)
plt.ylabel("Toplam Saatlerin Dağılımı", fontsize = size)

plt.xticks(range(1,25))
plt.grid(True)
plt.axhline(6500,linestyle = "--",color = "black")
plt.axhline(15000,linestyle ="--", color = "black")
plt.show()


# ***Hours Değişkeni İçin Score'lar Belirleyelim***
# 
# * 5,6,7,8,9,10,11 Yüksek Risk
# 
# * 3,4,12,19,20,21,22,23,24 Orta Risk
# 
# * 1,2,13,14,15,16,17,18 Düşük Risk

# In[44]:


def hours_risk(dataframe):
    
    hours = dataframe
    
    
    if hours in [5,6,7,8,9,10,11]:
        
        return "UpperRiskHours"
    
    elif hours in [3,4,12,19,20,21,22,23,24] :
        
        return "MiddleRiskHours"
    
    else:
        
        return "LowRiskHours"


# In[45]:


train["hours_risk"] = train["hours"].apply(hours_risk)
test["hours_risk"] = test["hours"].apply(hours_risk)


# In[46]:


train[train["hours_risk"] == "LowRiskHours"][["hours_risk","hours"]].head(200).T


# In[47]:


train["hours_risk"].value_counts()


# In[48]:


toplam = 0

for i in [5,6,7,8,9,10,11]:
    toplam+=train[train["hours"] == i]["hours"].count()

toplam    


# In[49]:


toplam = 0

for i in [3,4,12,19,20,21,22,23,24]:
    toplam+=train[train["hours"] == i]["hours"].count()

toplam


# In[50]:


toplam = 0

for i in [1,2,13,14,15,16,17,18]:
    toplam+=train[train["hours"] == i]["hours"].count()

toplam


# ## TransactionAMT

# In[51]:


describe = pd.DataFrame()

describe["TransactionAmt_Train"] = train["TransactionAmt"].describe()
describe["TransactionAmt_Test"] = test["TransactionAmt"].describe()

describe


# In[52]:


plt.figure(figsize = (15,10))
size = 15

plt.subplot(2,1,1)
sns.scatterplot(x = train["TransactionAmt"] , y = train["TransactionDT"], hue = train["isFraud"] )
plt.title("TransactionAmt vs TransactionDT For Train" , fontsize = size )
plt.xlabel("TransactionAmt", fontsize = size)
plt.ylabel("TransactionDT", fontsize = size)
plt.axvline(30000,linestyle = "--", color = "r")


plt.subplot(2,1,2)
sns.scatterplot(x = test["TransactionAmt"], y = test["TransactionDT"] )
plt.title("TransactionAmt vs TransactionDT For Test" , fontsize = size )
plt.xlabel("TransactionAmt", fontsize = size)
plt.ylabel("TransactionDT", fontsize = size)

plt.tight_layout()
plt.show()


# In[53]:


# Burda aynı işlem iki kare gerçekleşmiş !!!!


train[train["TransactionAmt"] > 30000]


# In[54]:


def check_outliers(dataframe, col_name, q1 = 0.25, q3 = 0.75, details = False):
    
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    
    interquantile = quantile3 - quantile1
    
    up_limit = quantile3 + 1.5 * interquantile
    low_limit = quantile1 - 1.5 * interquantile
    
    if details:
        
        up_limit = round(up_limit,4)
        low_limit = round(low_limit,4)
        
        print("Up Limit : {}\nLow Limit : {}".format(up_limit,low_limit))
    
    return up_limit, low_limit


# In[55]:


def grap_outliers(dataframe, col_name , q1 = 0.25, q3 = 0.75):
    
    up_limit, low_limit = check_outliers(dataframe, col_name , q1 , q3 ,details = False)
    
    outliers_data = (
            dataframe[ (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit) ].
            any(axis = None)
    )
    
    if outliers_data:
        
        print(f"{col_name.upper()} için Aykırı Değer Var !!!!")
        
    else:
        
        print(f"{col_name.upper()} için Aykırı Değer Yok !!!!")


# In[56]:


up_limit, low_limit = check_outliers(train,"TransactionAmt", details = True)


# In[57]:


grap_outliers(train , "TransactionAmt")


# In[58]:


# Burdaki Train için TransactionAmt benzer olan işlemleri çıkar !!!

train = train[~(train["TransactionAmt"] > 30000)]


# In[59]:


# Düzelecek !!!! Merak Etme !!!

describe["New_TransactionAmt_Train"] = train["TransactionAmt"].describe()

describe.T


# In[60]:


plt.figure(figsize = (15,15))

size = 15

plt.subplot(2,1,1)

sns.distplot(train[train["isFraud"] == 0 ]["TransactionAmt"], label = "NoFraud")
sns.distplot(train[train["isFraud"] == 1 ]["TransactionAmt"], label = "Fraud")

plt.xlabel("TransactionAmt" , fontsize = size)
plt.ylabel("Probability Density" , fontsize = size)
plt.title("TransactionAmt Distribution", fontsize = size) 
plt.legend(loc = "best",fontsize = size)



plt.subplot(2,1,2)

sns.distplot(test["TransactionAmt"] )


plt.xlabel("TransactionAmt" , fontsize = size)
plt.ylabel("Probability Density" , fontsize = size)
plt.title("TransactionAmt Distribution", fontsize = size) 

plt.show()


# In[61]:


train["LogTransactionAmt"] = np.log(train["TransactionAmt"])
test["LogTransactionAmt"] = np.log(test["TransactionAmt"])


# In[62]:


plt.figure(figsize = (15,15))

size = 15

plt.subplot(2,1,1)

sns.distplot(train[train["isFraud"] == 0 ]["LogTransactionAmt"], label = "NoFraud")
sns.distplot(train[train["isFraud"] == 1 ]["LogTransactionAmt"], label = "Fraud")

plt.xlabel("LogTransactionAmt" , fontsize = size)
plt.ylabel("Probability Density" , fontsize = size)
plt.title("LogTransactionAmt Distribution", fontsize = size) 
plt.legend(loc = "best",fontsize = size)



plt.subplot(2,1,2)

sns.distplot(test["LogTransactionAmt"] )


plt.xlabel("LogTransactionAmt" , fontsize = size)
plt.ylabel("Probability Density" , fontsize = size)
plt.title("LogTransactionAmt Distribution", fontsize = size) 

plt.show()


# ***Bu iki aralıkta Fraud olma olasılığı daha az iken;***
# 
# ***Diğer aralıklarda Fraud olma olasılığı daha yüksektir....***

# In[63]:


log_mean = train["LogTransactionAmt"].mean()
log_std = train["LogTransactionAmt"].std()

high, low = round(log_mean + log_std , 3) , round(log_mean - log_std, 3)

print("High Price : {}\nLow Price : {}".format(high, low))


# In[64]:


def Log_TransactionAmt_Risk(dataframe):
    
    log_transactionamt_risk = dataframe
    
    if log_transactionamt_risk < 5.318 and log_transactionamt_risk > 3.41:
        
        return "HighRisk"
    
    else:
        
        return "LowRisk"


# In[65]:


train["Risk_TransactionAmt"] = train["LogTransactionAmt"].apply(Log_TransactionAmt_Risk)
test["Risk_TransactionAmt"]  = test["LogTransactionAmt"].apply(Log_TransactionAmt_Risk)


# In[66]:


train[["Risk_TransactionAmt","LogTransactionAmt"]].head()


# In[67]:


train["Risk_TransactionAmt"].value_counts()


# ## ProductCD

# In[68]:


productCD_data= train.groupby("isFraud")["ProductCD"].value_counts(normalize = True).mul(100).rename("percentage").reset_index().sort_values("ProductCD")


plt.figure(figsize = (15,8))
productCD_bar = sns.barplot(x = "ProductCD", y = "percentage", data = productCD_data, hue = "isFraud", palette = "viridis")

for p in productCD_bar.patches:
    productCD_bar.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),                      ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points',size = size)


size = 15
plt.xlabel("ProductCD", fontsize = size)
plt.ylabel("Percentage", fontsize = size)
plt.title("ProductCD Fraud Yüzdelikleri",fontsize = size)
plt.legend(loc = "upper center",fontsize = size)
plt.xticks(fontsize = size)
plt.yticks(fontsize = size)
plt.show()


# In[69]:


def Risk_ProductCD(dataframe):
    
    productCD = dataframe
    
    if productCD == "C":
        
        return "HighRiskProductCD"
    
    elif productCD == "W":
    
        return "MiddleRiskProductCD"
    
    else:
        
        return "LowRiskProductCD"


# In[70]:


train["Risk_ProductCD"]= train["ProductCD"].apply(Risk_ProductCD)
test["Risk_ProductCD"] = test["ProductCD"].apply(Risk_ProductCD)


# In[71]:


train[["ProductCD","Risk_ProductCD"]].head(20).T


# ## Card

# In[72]:


train.loc[:,train.columns.str.contains("card")].head(5)


# In[73]:


data_card = [ "card"+str(i) for i in range(1,7)]

card_describe = pd.DataFrame()

card_num = []
card_cat = []

for col in data_card:
    
    if train[col].dtype != "object":
        
        card_num.append(col)
        
        card_describe[col] = train[col].describe()
    
    else:
        
        card_cat.append(col)
        
card_describe.T


# In[74]:


print("Card Numberic : {}\nCard Categoric : {}".format(card_num,card_cat) )


# In[75]:


plt.figure(figsize = (30,30))
num = 1

for col_name in card_num:
    
    plt.subplot(4,2,num)
    
    sns.distplot( train[(train["isFraud"] == 0) & (~train[col_name].isnull())][col_name]                   ,label = "NoFraud")
    
    sns.distplot( train[(train["isFraud"] == 1) & (~train[col_name].isnull())][col_name]                   ,label = "Fraud")
    
    plt.title(f"Train İçin {col_name.upper()} ", fontsize = size)
    plt.xlabel(f"{col_name.upper()}" , fontsize = size)
    plt.ylabel("Density" , fontsize = size)
    plt.legend(loc = "upper right")
    
    
    num+=1
    
    plt.subplot(4,2,num)
    
    sns.distplot(test[~(test[col_name].isnull())][col_name])
    
    plt.xlabel(f"{col_name.upper()}" , fontsize = size)
    plt.ylabel("Density" , fontsize = size)
    plt.title(f"Test İçin {col_name.upper()} ", fontsize = size)
    
    num+=1
    
plt.show()


# In[ ]:


plt.figure(figsize = (30,30))
num = 1

for col_name in card_num:
    
    plt.subplot(4,2,num)
    
    sns.distplot( train[(train["isFraud"] == 0)][col_name]                   ,label = "NoFraud")
    
    sns.distplot( train[(train["isFraud"] == 1)][col_name]                   ,label = "Fraud")
    
    plt.title(f"Train İçin {col_name.upper()} ", fontsize = size)
    plt.xlabel(f"{col_name.upper()}" , fontsize = size)
    plt.ylabel("Density" , fontsize = size)
    plt.legend(loc = "upper right")
    
    
    num+=1
    
    plt.subplot(4,2,num)
    
    sns.distplot(test[col_name])
    
    plt.xlabel(f"{col_name.upper()}" , fontsize = size)
    plt.ylabel("Density" , fontsize = size)
    plt.title(f"Test İçin {col_name.upper()} ", fontsize = size)
    
    num+=1
    
plt.show()


# In[76]:


for col in data_card:
    
    print("{} ---> NaN Count : {}".format(col, train[col].isnull().sum()))


# In[77]:


train[train["card3"] > 160]["isFraud"].value_counts()


# In[78]:


train[train["card3"] <= 160]["isFraud"].value_counts()


# In[79]:


# Positive ---- > 1 Fraud
# Negative ---- > 0 No Fraud


def card3_create(dataframe):
    
    card3 = dataframe
    
    if card3 > 160 :
        return "Positive"
    
    elif card3 <= 160 :
        return "Negative"
    
    else:
        return "UnKnown"


# In[80]:


train["card3_feature"] = train["card3"].apply(card3_create)
test["card3_feature"] = test["card3"].apply(card3_create)


# In[81]:


train["card3_feature"].value_counts()


# In[82]:


test["card3_feature"].value_counts()


# In[83]:


np.nan > 1 , np.nan < -1, np.nan == 1


# In[84]:


card_cat


# In[85]:


#fig, axes = plt.subplots(2,1,figsize = (12, 12))

train_card4 = (
    train[~(train["card4"].isnull())].groupby("isFraud")["card4"].value_counts(normalize = True).
    rename("percentage").mul(100).reset_index().sort_values("card4")

)


# In[86]:


train[train["isFraud"] == 0]["card4"].value_counts() / len(train[train["isFraud"] == 0]) * 100


# In[87]:


train_card4[train_card4["isFraud"] == 0]


# In[88]:


pd.DataFrame(((train.groupby("card4")["isFraud"].value_counts() / len(train)) * 100)).reset_index(0)


# In[92]:


train_card4 = (
    train[~(train["card4"].isnull())].groupby("isFraud")["card4"].value_counts(normalize = True).
    rename("percentage").mul(100).reset_index().sort_values("card4")
    )

test_card4 = (
    test[~(test["card4"].isnull())]["card4"].value_counts(normalize = True).mul(100).
    rename("percentage").reset_index()
    )

test_card4.columns = ["card4","Percentage"]
test_card4 = test_card4.sort_values("card4")


fig, axes = plt.subplots(2,1,figsize = (15,15))

card4_train_bar = sns.barplot(x = "card4", y = "percentage", data = train_card4, 
                              hue = "isFraud",ax= axes[0],palette = "viridis")

card4_test_bar = sns.barplot(x = "card4" , y = "Percentage", data = test_card4, ax = axes[1],
                             palette = "viridis")

for p in card4_test_bar.patches:
    card4_test_bar.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),                      ha = 'center', va = 'center', 
                     xytext = (0, 5), textcoords = 'offset points',fontsize = 15)

for p in card4_train_bar.patches:
    card4_train_bar.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),                      ha = 'center', va = 'center',
                     xytext = (0, 5), textcoords = 'offset points',fontsize = 15)
    
size = 15
axes[0].set_title("Train İçin Kart Tiplerine Göre Fraud ve NoFraud Oranları", fontsize = size)
axes[0].set_xlabel("Card4", fontsize = size)
axes[0].set_ylabel("Percentage", fontsize = size)


axes[1].set_title("Test İçin Kart Tiplerine Göre Fraud ve NoFraud Oranları", fontsize = size)
axes[1].set_xlabel("Card4", fontsize = size)
axes[1].set_ylabel("Percentage", fontsize = size)


plt.show()


# In[93]:


train_card6 = (
    train[~(train["card6"].isnull())].groupby("isFraud")["card6"].value_counts(normalize = True).
    rename("percentage").mul(100).reset_index().sort_values("card6")
    )

test_card6 = (
    test[~(test["card6"].isnull())]["card6"].value_counts(normalize = True).mul(100).
    rename("percentage").reset_index()
    )

test_card6.columns = ["card6","Percentage"]
test_card6 = test_card6.sort_values("card6")


fig, axes = plt.subplots(2,1,figsize = (15,15))

card6_train_bar = sns.barplot(x = "card6", y = "percentage", data = train_card6, 
                              hue = "isFraud",ax= axes[0],palette="viridis")

card6_test_bar = sns.barplot(x = "card6" , y = "Percentage", data = test_card6, ax = axes[1],
                             palette="viridis")

for p in card6_test_bar.patches:
    card6_test_bar.annotate(format(p.get_height(), '.5f'), (p.get_x() + p.get_width() / 2., p.get_height()),                      ha = 'center', va = 'center', 
                     xytext = (0, 5), textcoords = 'offset points',fontsize = 15)

for p in card6_train_bar.patches:
    card6_train_bar.annotate(format(p.get_height(), '.5f'), (p.get_x() + p.get_width() / 2., p.get_height()),                      ha = 'center', va = 'center',
                     xytext = (0, 5), textcoords = 'offset points',fontsize = 15)
    
size = 15
axes[0].set_title("Train İçin Kart Tiplerine Göre Fraud ve NoFraud Oranları", fontsize = size)
axes[0].set_xlabel("Card6", fontsize = size)
axes[0].set_ylabel("Percentage", fontsize = size)


axes[1].set_title("Test İçin Kart Tiplerine Göre Fraud ve NoFraud Oranları", fontsize = size)
axes[1].set_xlabel("Card6", fontsize = size)
axes[1].set_ylabel("Percentage", fontsize = size)


plt.show()


# In[94]:


print("Train Verisi İçin %s Tane Charge Card Gözlem Vardır ..." 
      % train[train["card6"] == "charge card"].shape[0])

print("Train Verisi İçin %s Tane Debit or Credit Gözlem Vardır ..." 
      % train[train["card6"] == "debit or credit"].shape[0])


# In[95]:


for col in data_card:
    print(f"{col} ---> {train[col].isnull().sum()}")


# In[96]:


def replace_card(dataframe):
    
    card6 = dataframe
    
    if card6 == "charge card" or card6 == "debit or credit":
        
        return "debit"
    
    else:
        
        return card6


# In[97]:


train["card6"] = train["card6"].fillna("UnKnown")
test["card6"] = test["card6"].fillna("UnKnown")


# In[98]:


train["card6"] = train["card6"].apply(replace_card)
test["card6"] = test["card6"].apply(replace_card)


# In[99]:


train["card6"].unique(),test["card6"].unique()


# In[100]:


train_card6 = (
    train[~(train["card6"].isnull())].groupby("isFraud")["card6"].value_counts(normalize = True).
    rename("percentage").mul(100).reset_index().sort_values("card6")
    )

test_card6 = (
    test[~(test["card6"].isnull())]["card6"].value_counts(normalize = True).mul(100).
    rename("percentage").reset_index()
    )

test_card6.columns = ["card6","Percentage"]
test_card6 = test_card6.sort_values("card6")


fig, axes = plt.subplots(2,1,figsize = (15,15))

card6_train_bar = sns.barplot(x = "card6", y = "percentage", data = train_card6, 
                              hue = "isFraud",ax= axes[0])

card6_test_bar = sns.barplot(x = "card6" , y = "Percentage", data = test_card6, ax = axes[1])

for p in card6_test_bar.patches:
    card6_test_bar.annotate(format(p.get_height(), '.5f'), (p.get_x() + p.get_width() / 2., p.get_height()),                      ha = 'center', va = 'center', 
                     xytext = (0, 5), textcoords = 'offset points',fontsize = 15)

for p in card6_train_bar.patches:
    card6_train_bar.annotate(format(p.get_height(), '.5f'), (p.get_x() + p.get_width() / 2., p.get_height()),                      ha = 'center', va = 'center',
                     xytext = (0, 5), textcoords = 'offset points',fontsize = 15)
    
size = 15
axes[0].set_title("Train İçin Kart Tiplerine Göre Fraud ve NoFraud Oranları", fontsize = size)
axes[0].set_xlabel("Card6", fontsize = size)
axes[0].set_ylabel("Percentage", fontsize = size)


axes[1].set_title("Test İçin Kart Tiplerine Göre Fraud ve NoFraud Oranları", fontsize = size)
axes[1].set_xlabel("Card6", fontsize = size)
axes[1].set_ylabel("Percentage", fontsize = size)


plt.show()


# ## addr1 and addr2

# In[101]:


fig , axes = plt.subplots(2,1, figsize = (12,12))

sns.distplot(train[(train["isFraud"] == 0) & (~train["addr1"].isnull())]["addr1"], 
             ax = axes[0],label = "NoFraud",hist_kws={"edgecolor" : "black"})
sns.distplot(train[(train["isFraud"] == 1) & (~train["addr1"].isnull())]["addr1"],
             ax = axes[0],label = "Fraud",hist_kws={"edgecolor" : "black"})


sns.distplot(train[(train["isFraud"] == 0) & (~train["addr2"].isnull())]["addr2"],
             ax = axes[1],label = "NoFraud",hist_kws={"edgecolor" : "black"})
sns.distplot(train[(train["isFraud"] == 1) & (~train["addr2"].isnull())]["addr2"],
             ax = axes[1],label = "Fraud",hist_kws={"edgecolor" : "black"})

size = 15
axes[0].set_ylabel("Probability Density".upper() , fontsize = size)
axes[0].set_xlabel("addr1".upper(), fontsize = size)
axes[0].set_title("Purchaser Region in Train".upper() , fontsize = size)
axes[0].legend(loc = "best")
axes[0].set_xticks(range(1,600,20))

axes[1].set_ylabel("Probability Density".upper() , fontsize = size)
axes[1].set_xlabel("addr1".upper(), fontsize = size)
axes[1].set_title("Purchaser Region in Train".upper() , fontsize = size)
axes[1].legend(loc = "best")
axes[1].set_xticks(range(1,102,2))

plt.tight_layout()
plt.show()


# In[102]:


# addr2 için yeterli bir bilgi sahibi değiliz !!!!


def addr1_Risk(dataframe):
    
    addr1 = dataframe
    
    if addr1 < 340 and addr1 > 320:
        
        return "Addr1Risk"
    
    else:
        
        return "NoInf"


# In[103]:


train["addr1_risk"] = train["addr1"].apply(addr1_Risk)
test["addr1_risk"] = test["addr1"].apply(addr1_Risk)


# ## dist1 and dist2

# In[104]:


gc.collect()


# In[105]:


dist_describe = pd.DataFrame()

dist_describe["Train_dist1"] = train["dist1"].describe()
dist_describe["Train_dist2"] = train["dist2"].describe()

dist_describe["Test_dist1"] = test["dist1"].describe()
dist_describe["Test_dist2"] = test["dist2"].describe()

dist_describe.T


# In[106]:


size = 15
plt.figure(figsize = (15,15))

plt.subplot(2,1,1)

sns.distplot(train[(train["isFraud"] == 0) & (~train["dist1"].isnull())]["dist1"],
            hist_kws={"edgecolor":"black"})
sns.distplot(train[(train["isFraud"] == 1) & (~train["dist1"].isnull())]["dist1"],
             hist_kws={"edgecolor":"black"})
               
    
plt.title("Dist1 Distance In Train", fontsize = size)
plt.xlabel("Dist1".upper() , fontsize = size)
plt.ylabel("Probability".upper(), fontsize = size)
plt.legend(["NoFraud","Fraud"],loc = "best",fontsize = size)
                   
                   
plt.subplot(2,1,2)
                   
sns.distplot(train[(train["isFraud"] == 0) & (~train["dist2"].isnull())]["dist2"],
            hist_kws={"edgecolor":"black"})
sns.distplot(train[(train["isFraud"] == 1) & (~train["dist2"].isnull())]["dist2"],
            hist_kws={"edgecolor":"black"})                
                   
                   
plt.title("Dist2 Distance In Train", fontsize = size)
plt.xlabel("Dist2".upper() , fontsize = size)
plt.ylabel("Probability".upper(), fontsize = size)
plt.legend(["NoFraud","Fraud"],loc = "best",fontsize = size)
                   
plt.tight_layout()
plt.show()


# In[107]:


def dist1_risk(dataframe):
    
    dist1 = dataframe
    
    if dist1 < 2000:
        
        return "HighDist1Risk"
    
    else:
        
        return "LowDist1Risk"
    
def dist2_risk(dataframe):
    
    dist2 = dataframe
    
    if dist2 < 3000:
        
        return "HighDist2Risk"
    
    else:
        
        return "LowDist2Risk"


# In[108]:


train["Dist1_Risk"] = train["dist1"].apply(dist1_risk)
test["Dist1_Risk"] = test["dist1"].apply(dist1_risk)

train["Dist2_Risk"] = train["dist2"].apply(dist2_risk)
test["Dist2_Risk"] = test["dist2"].apply(dist2_risk)


# In[109]:


train["Dist1_Risk"].isnull().sum(),train["Dist2_Risk"].isnull().sum()


# In[110]:


gc.collect()


# In[111]:


fig, axes = plt.subplots(2, 1 , figsize = (15,15))

sns.scatterplot(x = "TransactionAmt", y = "dist1", data = train[~train["dist1"].isnull()],
                alpha = .9, hue = "isFraud", ax = axes[0])

sns.scatterplot(x = "TransactionAmt", y = "dist2", data = train[~train["dist2"].isnull()],
                alpha = .9, hue = "isFraud", ax = axes[1])

size = 15

axes[0].set_title("Fraud & NoFraud için TranscationAmt Ve Dist1 Dağılımı" , fontsize = size)
axes[1].set_title("Fraud & NoFraud için TranscationAmt Ve Dist1 Dağılımı" , fontsize = size)

#axes[0].legend(fontsize = size)
#axes[1].legend(fontsize = size)

axes[0].set_xlabel("TransactionAmt", fontsize = size)
axes[1].set_xlabel("TransactionAmt", fontsize = size)

axes[0].set_ylabel("Dist1", fontsize = size)
axes[1].set_ylabel("Dist2", fontsize = size)

plt.tight_layout()
plt.show()


# In[112]:


print("Before Train Shape : {}".format(train.shape))
train = train[~(train["dist1"] > 6000)]
train = train[~(train["dist1"] > 8000)]
print("After Train Shape : {}".format(train.shape))
gc.collect()


# ## P_EmailDomain and R_EmailDomain

# In[113]:


print("P_EmailDomain Nan Values : {}".format(train["P_emaildomain"].isnull().sum()))
print("R_EmailDomain Nan Values : {}".format(train["R_emaildomain"].isnull().sum()))


# In[114]:


train["P_emaildomain"].fillna("UnKnown", inplace  = True)
train["R_emaildomain"].fillna("UnKnown", inplace  = True)

test["P_emaildomain"].fillna("UnKnown", inplace  = True)
test["R_emaildomain"].fillna("UnKnown", inplace  = True)

print("Train P_EmailDomain Nan Values : {}".format(train["P_emaildomain"].isnull().sum()))
print("Train R_EmailDomain Nan Values : {}".format(train["R_emaildomain"].isnull().sum()))

print("Test P_EmailDomain Nan Values : {}".format(test["P_emaildomain"].isnull().sum()))
print("Test R_EmailDomain Nan Values : {}".format(test["R_emaildomain"].isnull().sum()))


# In[115]:


plt.figure(figsize = (15,15))

R_email = train["R_emaildomain"].value_counts().reset_index()
R_email.columns = ["Name", "Count"]
R_email = R_email.sort_values(by = "Count", ascending = False)

P_email = train["P_emaildomain"].value_counts().reset_index()
P_email.columns = ["Name", "Count"]
P_email = P_email.sort_values(by = "Count", ascending = False)

size = 15

plt.subplot(2,1,1) 

sns.barplot(x = "Name", y = "Count", data = R_email)
plt.xticks(rotation = 90)
plt.xlabel("Name", fontsize = size)
plt.ylabel("Count", fontsize = size)
plt.title("R_EmailDomain Count", fontsize = size)

plt.subplot(2,1,2) 

sns.barplot(x = "Name", y = "Count", data = P_email)
plt.xticks(rotation = 90)
plt.xlabel("Name", fontsize = size)
plt.ylabel("Count", fontsize = size)
plt.title("P_EmailDomain Count", fontsize = size)

plt.tight_layout()
plt.show()


# In[116]:


plt.figure(figsize = (45,45))


P_email_Fraud = (
    
    train.groupby("isFraud")["P_emaildomain"].value_counts(normalize = True).rename("percantage").
    mul(100).reset_index().sort_values("percantage")
)


E_email_Fraud = (
    
    train.groupby("isFraud")["R_emaildomain"].value_counts(normalize = True).rename("percantage").
    mul(100).reset_index().sort_values("percantage")
)


size = 15

plt.subplot(2,1,1) 

sns.barplot(x = "P_emaildomain", y = "percantage", data = P_email_Fraud,hue = "isFraud",
           palette="viridis")
plt.xticks(rotation = 90)
plt.xlabel("Name", fontsize = size)
plt.ylabel("Percentage", fontsize = size)
plt.title("P_EmailDomain_Fraud", fontsize = size)

plt.subplot(2,1,2) 

sns.barplot(x = "R_emaildomain", y = "percantage", data = E_email_Fraud, hue = "isFraud",
           palette="viridis")
plt.xticks(rotation = 90)
plt.xlabel("Name", fontsize = size)
plt.ylabel("Percentage", fontsize = size)
plt.title("R_EmailDomain_Fraud", fontsize = size)

plt.tight_layout()
plt.show()


# In[117]:


train["P_emaildomain"].unique()


# In[118]:


gc.collect()


# In[119]:


train[train["P_emaildomain"].str.contains("mac")]["P_emaildomain"].unique()


# In[120]:


train.loc[:,"P_emaildomain"] = np.where(train["P_emaildomain"].isin(["gmail.com","gmail"]), 
                                        "Google", train["P_emaildomain"])

train.loc[:,"P_emaildomain"] = np.where(train["P_emaildomain"].isin(['yahoo.com', 'yahoo.com.mx',
                                                                     'yahoo.fr', 'yahoo.de', 
                                                                     'yahoo.es','yahoo.co.uk', 
                                                                     'yahoo.co.jp']), "Yahoo",
                                                                      train["P_emaildomain"])

train.loc[:,"P_emaildomain"] = np.where(train["P_emaildomain"].isin(['hotmail.com', 'hotmail.es', 
                                                                      'hotmail.fr', 'hotmail.de',
                                                                      'hotmail.co.uk','live.com.mx', 
                                                                      'live.com', 'live.fr','msn.com',
                                                                      'outlook.com', 'outlook.es']), 
                                                                      "Hotmail",train["P_emaildomain"])

train.loc[:,"P_emaildomain"] = np.where(train["P_emaildomain"].isin(['mac.com','icloud.com','me.com']),
                                                                   "Apple",train["P_emaildomain"])

less_500 = pd.DataFrame(train["P_emaildomain"].value_counts()).reset_index()
less_500.columns = ["Name","Count"]

list_500 = less_500[less_500["Count"] < 500]["Name"].values.tolist()

train.loc[:,"P_emaildomain"] = np.where(train["P_emaildomain"].isin(list_500), 
                                        "Others", train["P_emaildomain"])


# In[121]:


test.loc[:,"P_emaildomain"] = np.where(test["P_emaildomain"].isin(["gmail.com","gmail"]), 
                                        "Google", test["P_emaildomain"])

test.loc[:,"P_emaildomain"] = np.where(test["P_emaildomain"].isin(['yahoo.com', 'yahoo.com.mx',
                                                                     'yahoo.fr', 'yahoo.de', 
                                                                     'yahoo.es','yahoo.co.uk', 
                                                                     'yahoo.co.jp']), "Yahoo",
                                                                      test["P_emaildomain"])

test.loc[:,"P_emaildomain"] = np.where(test["P_emaildomain"].isin(['hotmail.com', 'hotmail.es', 
                                                                      'hotmail.fr', 'hotmail.de',
                                                                      'hotmail.co.uk','live.com.mx', 
                                                                      'live.com', 'live.fr','msn.com',
                                                                      'outlook.com', 'outlook.es']), 
                                                                      "Hotmail",test["P_emaildomain"])

test.loc[:,"P_emaildomain"] = np.where(test["P_emaildomain"].isin(['mac.com','icloud.com','me.com']),
                                                                   "Apple",test["P_emaildomain"])

less_500 = pd.DataFrame(test["P_emaildomain"].value_counts()).reset_index()
less_500.columns = ["Name","Count"]

list_500 = less_500[less_500["Count"] < 500]["Name"].values.tolist()

test.loc[:,"P_emaildomain"] = np.where(test["P_emaildomain"].isin(list_500), 
                                        "Others", test["P_emaildomain"])


# In[122]:


train.loc[:,"R_emaildomain"] = np.where(train["R_emaildomain"].isin(["gmail.com","gmail"]), 
                                        "Google", train["R_emaildomain"])

train.loc[:,"R_emaildomain"] = np.where(train["R_emaildomain"].isin(['yahoo.com', 'yahoo.com.mx',
                                                                     'yahoo.fr', 'yahoo.de', 
                                                                     'yahoo.es','yahoo.co.uk', 
                                                                     'yahoo.co.jp']), "Yahoo",
                                                                      train["R_emaildomain"])

train.loc[:,"R_emaildomain"] = np.where(train["R_emaildomain"].isin(['hotmail.com', 'hotmail.es', 
                                                                      'hotmail.fr', 'hotmail.de',
                                                                      'hotmail.co.uk','live.com.mx', 
                                                                      'live.com', 'live.fr','msn.com',
                                                                      'outlook.com', 'outlook.es']), 
                                                                      "Hotmail",train["R_emaildomain"])

train.loc[:,"R_emaildomain"] = np.where(train["R_emaildomain"].isin(['mac.com','icloud.com','me.com']),
                                                                   "Apple",train["R_emaildomain"])

less_500 = pd.DataFrame(train["R_emaildomain"].value_counts()).reset_index()
less_500.columns = ["Name","Count"]

list_500 = less_500[less_500["Count"] < 500]["Name"].values.tolist()

train.loc[:,"R_emaildomain"] = np.where(train["R_emaildomain"].isin(list_500), 
                                        "Others", train["R_emaildomain"])


# In[123]:


test.loc[:,"R_emaildomain"] = np.where(test["R_emaildomain"].isin(["gmail.com","gmail"]), 
                                        "Google", test["R_emaildomain"])

test.loc[:,"R_emaildomain"] = np.where(test["R_emaildomain"].isin(['yahoo.com', 'yahoo.com.mx',
                                                                     'yahoo.fr', 'yahoo.de', 
                                                                     'yahoo.es','yahoo.co.uk', 
                                                                     'yahoo.co.jp']), "Yahoo",
                                                                      test["R_emaildomain"])

test.loc[:,"R_emaildomain"] = np.where(test["R_emaildomain"].isin(['hotmail.com', 'hotmail.es', 
                                                                      'hotmail.fr', 'hotmail.de',
                                                                      'hotmail.co.uk','live.com.mx', 
                                                                      'live.com', 'live.fr','msn.com',
                                                                      'outlook.com', 'outlook.es']), 
                                                                      "Hotmail",test["R_emaildomain"])

test.loc[:,"R_emaildomain"] = np.where(test["R_emaildomain"].isin(['mac.com','icloud.com','me.com']),
                                                                   "Apple",test["R_emaildomain"])

less_500 = pd.DataFrame(test["R_emaildomain"].value_counts()).reset_index()
less_500.columns = ["Name","Count"]

list_500 = less_500[less_500["Count"] < 500]["Name"].values.tolist()

test.loc[:,"R_emaildomain"] = np.where(test["R_emaildomain"].isin(list_500), 
                                        "Others", test["R_emaildomain"])


# In[124]:


test["R_emaildomain"].unique()


# In[126]:


plt.figure(figsize = (30,30))


P_email_Fraud = (
    
    train.groupby("isFraud")["P_emaildomain"].value_counts(normalize = True).rename("percantage").
    mul(100).reset_index().sort_values("percantage")
)


E_email_Fraud = (
    
    train.groupby("isFraud")["R_emaildomain"].value_counts(normalize = True).rename("percantage").
    mul(100).reset_index().sort_values("percantage")
)


size = 20

plt.subplot(2,1,1) 

sns.barplot(x = "P_emaildomain", y = "percantage", data = P_email_Fraud,hue = "isFraud",palette="viridis")
plt.xticks(fontsize = size,rotation = 45)
plt.xlabel("Name", fontsize = size)
plt.ylabel("Percentage", fontsize = size)
plt.title("P_EmailDomain_Fraud", fontsize = size)

plt.subplot(2,1,2) 

sns.barplot(x = "R_emaildomain", y = "percantage", data = E_email_Fraud, hue = "isFraud",palette="viridis")
plt.xticks(fontsize = size,rotation = 45)
plt.xlabel("Name", fontsize = size)
plt.ylabel("Percentage", fontsize = size)
plt.title("R_EmailDomain_Fraud", fontsize = size)

plt.tight_layout()
plt.show()


# In[127]:


gc.collect()


# ## C1-C14

# In[128]:


train.loc[:, train.columns.str.contains("C")].head(3)


# In[111]:


plt.figure(figsize = (20,50))

data_C = ["C"+str(i) for i in range(1,15)]
i = 1
size = 15

for col in data_C:
    
    plt.subplot(14,2,i)
    sns.scatterplot( x = "TransactionDT", y = col, hue = "isFraud",data = train[~(train[col].isnull())] )
    plt.title(f"Train {col.upper()}", fontsize = size)
    
    i+=1
    
    plt.subplot(14,2,i)
    
    sns.scatterplot( x = "TransactionDT", y = col, data = test[~(test[col].isnull())] )
    plt.title(f"Test {col.upper()}", fontsize = size)
    
    i+=1
    
plt.tight_layout()
plt.show()


# In[129]:


train = train[~(train["C1"] > 3000)]
train = train[~(train["C2"] > 3000)]
train = train[~(train["C4"] > 1400)]
train = train[~(train["C6"] > 1400)]
train = train[~(train["C7"] > 1400)]
train = train[~(train["C8"] > 2000)]
train = train[~(train["C10"] > 2100)]
train = train[~(train["C11"] > 2000)]
train = train[~(train["C12"] > 2000)]
train = train[~(train["C13"] > 1950)]
train = train[~(train["C14"] > 900)]


# In[131]:


data_C = ["C"+str(i) for i in range(1,15)]

for c in data_C:
    print(f"{c} Train için {train[c].isnull().sum()} değer boştur !!!")
    print(f"{c} Test için {test[c].isnull().sum()} değer boştur !!!",end = "\n\n")


# ## D1-D15

# In[132]:


data_D = ["D"+str(i) for i in range(1,16)]


# In[133]:


for d in data_D:
    print(f"{d} Train için %{train[d].isnull().sum() / train.shape[0] * 100} değer boştur !!!")
    print(f"{d} Test için %{test[d].isnull().sum() / train.shape[0] * 100} değer boştur !!!",end = "\n\n")


# In[120]:


fig, axes = plt.subplots(15,2,figsize = (20,80))
size = 15
i=0

for col in data_D:
    
    sns.scatterplot(x = "TransactionDT", y = col, hue = "isFraud", 
                    data = train[~(train[col].isnull())] ,ax = axes[i][0])
    
    axes[i][0].set_title(f"Train {col.upper()}",fontsize = size)
    #axes[i][0].legend(loc = "best")
    
    sns.scatterplot(x = "TransactionDT", y = col , 
                    data = test[~(test[col].isnull())] ,ax = axes[i][1])
    
    axes[i][1].set_title(f"Train {col.upper()}",fontsize = size)
    #axes[i][1].legend(loc = "best")
    
    i+=1
    
plt.tight_layout()
plt.show()


# - D1-15 : Önceki Işlem Arasındaki Günler Vb. Gibi Zaman Deltası
# - Burda TransactionDT arttığında D özelliğinin arttığını gözlemlemekteyiz...
# - Herhangi bir aykırı bir değer var varsa da görmezden gelenebilir ...
# - Her biri için ( D1-15) %40 fazla eksik değer vardır..

# In[134]:


gc.collect()


# ## M1 - M9
# 
# - Karttaki isimler ve Adres vb. Gibi Eşleme

# In[135]:


data_M = ["M"+str(i) for i in range(1,10)]

for m in data_M:
    print(f"{m} Train için % {(train[m].isnull().sum() / len(train )*100)}")
    print(f"{m} Test için % {(test[m].isnull().sum() / len(test)) *100}",end= "\n\n")


# In[136]:


train[data_M].head(3)


# In[137]:


plt.figure(figsize = (15,15))
i = 1

for m in data_M:
    
    m_train = train.groupby("isFraud")[m].value_counts(normalize = True).rename("percentage").mul(100).              reset_index().sort_values(m)
    
    plt.subplot(3,3,i)
    m_bar = sns.barplot(x = m , y = "percentage", data = m_train, palette = "viridis", hue = "isFraud")
    plt.title(f"{m.upper()}",fontsize = 15)
    
    for p in m_bar.patches:
        m_bar.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    
    i+=1
    
plt.tight_layout()
plt.show()


# ## V Features
# 
# - Vesta, sıralama, sayma ve diğer varlık ilişkileri dahil olmak üzere zengin özellikler tasarladı.

# In[138]:


gc.collect()


# In[139]:


data_V = ["V" + str(i) for i in range(1,340) if i != 107]
data_V.append("TransactionDT")

#plt.figure(figsize = (15,15))
#sns.heatmap(train[data_V].corr(), cmap = "viridis", annot = False, center = 0.0)
#plt.title("V1-V339", fontsize = size)
#plt.show()


# In[140]:


gc.collect()


# ## id01-id38

# In[141]:


data_ID = train.loc[:,train.columns.str.contains("id")].columns.tolist()

for ID in data_ID:
    
    if train[ID].dtype == "object":
        
        print(f"{ID} : {train[ID].unique()}")    


# In[142]:


train["id_30"].fillna("UnKnown", inplace = True)
test["id-30"].fillna("UnKnown",inplace = True)

train["OS_id_30"] = train["id_30"].replace("[\d\s\_.]", "", regex=True)
train["Version_id_30"] = train["id_30"].replace("[\_]", ".", 
                                                regex = True).replace("[A-z]","",regex = True)

test["OS_id_30"] = test["id-30"].replace("[\d\s\_.]", "", regex=True)
test["Version_id_30"] = test["id-30"].replace("[\_]", ".", 
                                                regex = True).replace("[A-z]","",regex = True)

train["Version_id_30"] = train["Version_id_30"].replace("","No.Version")
test["Version_id_30"] = test["Version_id_30"].replace("", "No.Version")


# In[143]:


test[["OS_id_30","id-30","Version_id_30"]].isnull().sum()


# In[144]:


train[["OS_id_30","id_30","Version_id_30"]].isnull().sum()


# In[145]:


train["id_31"].fillna("UnKnown", inplace = True)
test["id-31"].fillna("UnKnown",inplace = True)

train["id_31"].replace("/",".",regex = True, inplace = True)
test["id-31"].replace("/",".",regex = True, inplace = True)


# In[146]:


train["Browser_id_31"] = train["id_31"].replace("[\d.-]","",regex = True)
test["Browser_id_31"] = test["id-31"].replace("[\d.-]","",regex = True)

train["Version_id_31"] = train["id_31"].replace("[A-z-]","",regex = True)
test["Version_id_31"] = test["id-31"].replace("[A-z-]","",regex = True)


# In[147]:


train["Version_id_31"] = train["Version_id_31"].replace("","No.Version")
test["Version_id_31"] = test["Version_id_31"].replace("","No.Version")


# In[148]:


train[["id_31","Version_id_31","Browser_id_31"]].isnull().sum()


# In[149]:


test[["id-31","Version_id_31","Browser_id_31"]].isnull().sum()


# In[150]:


train["id_33"].fillna("unknownxunknown",inplace = True)
test["id-33"].fillna("unknownxunknown",inplace = True)

train["Width_id_33"] = train["id_33"].str.split("x", expand = True)[0]
train["Height_id_33"] = train["id_33"].str.split("x", expand = True)[1]

test["Width_id_33"] = test["id-33"].str.split("x", expand = True)[0]
test["Height_id_33"] = test["id-33"].str.split("x", expand = True)[1]


# In[151]:


train[["id_33","Width_id_33","Height_id_33"]].isnull().sum()


# In[152]:


test[["id-33","Width_id_33","Height_id_33"]].isnull().sum()


# In[153]:


cat_ID = [col for col in data_ID if train[col].dtype == "object"]

i = 1

plt.figure(figsize = (15,20))

for col in cat_ID:
    
    plt.subplot(5,3,i)
    
    train_id = train.groupby("isFraud")[col].value_counts(normalize = True).rename("percentage").mul(100).               reset_index().sort_values(col)
    
    bar_id =sns.barplot(x = col , y = "percentage", data = train_id, hue = "isFraud",palette = "viridis")
    plt.title(f"{col.upper()}")
    plt.legend(loc = "best")
    #plt.xticks(rotation = 45)
    
    for p in bar_id.patches:
        bar_id.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    
    i+=1
    
plt.tight_layout()
plt.show()


# ## DeviceType and DeviceInfo

# In[154]:


train[["DeviceType","DeviceInfo"]].head()


# In[155]:


plt.figure(figsize = (15,8))
size = 15

plt.subplot(1,2,1)

train_type = (
    train[~(train["DeviceType"].isnull())].groupby("isFraud")["DeviceType"].value_counts(normalize = True)
    .rename("percentage").mul(100).reset_index().sort_values("DeviceType")
)

sns.barplot(x = "DeviceType", y = "percentage", data = train_type , hue="isFraud", palette = "viridis")
plt.xlabel("DeviceType", fontsize = size)
plt.ylabel("Percentage", fontsize = size )
plt.title("Train DeviceType" , fontsize = size)


plt.subplot(1,2,2)

test_type = (
    test[~(test["DeviceType"].isnull())][["DeviceType"]].value_counts(normalize = True).mul(100).
    rename("percentage").reset_index()
    
)

sns.barplot(x = "DeviceType", y = "percentage", data = test_type, palette = "viridis")
plt.xlabel("DeviceType", fontsize = size)
plt.ylabel("Percentage", fontsize = size )
plt.title("Test DeviceType" , fontsize = size)

plt.tight_layout()
plt.show()


# In[156]:


# DeviceType
train["DeviceType"].fillna("UnKnown",inplace = True)
test["DeviceType"].fillna("UnKnown", inplace = True)


# In[157]:


# DeviceInfo

train["DeviceInfo"].fillna("UnKnown",inplace = True)
test["DeviceInfo"].fillna("UnKnown",inplace = True)


# In[158]:


print("DeviceInfo Nan Values Train : {}".format(train["DeviceInfo"].isnull().sum()))
print("DeviceInfo Nan Values Test  : {}".format(test["DeviceInfo"].isnull().sum()))


# In[159]:


train["DeviceInfo"].value_counts()


# In[160]:


def device_transform(data):
    
    data["DeviceCorp"] = data["DeviceInfo"]
    
    #data.loc[:, "DeviceCorp"] = data["DeviceInfo"].str.split("/", expand= True)[0]
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("rv:",na = False) , 
                                         "RV" , data["DeviceCorp"])    
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("SM-",na = False) , 
                                         "SAMSUNG" , data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("SAMSUNG",na = False) , 
                                         "SAMSUNG" , data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("GT-",na = False) , 
                                         "SAMSUNG" , data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("HONOR",na = False) , 
                                         "HUAWEI" , data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("ALE-",na = False) , 
                                         "HUAWEI" , data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("-L",na = False) , 
                                         "HUAWEI" , data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("iOS",na = False) , 
                                         "APPLE" , data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("MacOS",na = False) , 
                                         "APPLE" , data["DeviceCorp"])

    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("Idea",na = False) , 
                                         "LENOVO" , data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("TA",na = False) , 
                                         "LENOVO", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("Moto",na = False) , 
                                         "MOTO", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("moto",na = False) , 
                                         "MOTO", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("Moto G",na = False) , 
                                         "MOTO", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("XT",na = False) , 
                                         "MOTO", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("Edison",na = False) , 
                                         "MOTO", data["DeviceCorp"])
        
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("MI",na = False) , 
                                         "XIAOMI", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("Mi",na = False) , 
                                         "XIAOMI", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("Redmi",na = False) , 
                                         "XIAOMI", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("VS",na = False) , 
                                         "LG", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("LG",na = False) , 
                                         "LG", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("EGO",na = False) , 
                                         "LG", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("ALCATEL",na = False) , 
                                         "ALCATEL", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("ONE TOUCH",na = False) , 
                                         "ALCATEL", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("ONE A",na = False) , 
                                         "ONEPLUS", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("OPR6",na = False) , 
                                         "HTC", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("HTC",na = False) , 
                                         "HTC", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("Nexus",na = False) , 
                                         "GOOGLE", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("Pixel",na = False) , 
                                         "GOOGLE", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("STV",na = False) , 
                                         "BLACKBERRY", data["DeviceCorp"])

    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("ASUS",na = False) , 
                                         "ASUS", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("BLADE",na = False) , 
                                         "ZTE", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.contains("Blade",na = False) , 
                                         "ZTE", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].isin(["rv","SM","GT","SGH"]),
                                         "SAMSUNG", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.startswith("Z",na = False), 
                                         "ZTE", data["DeviceCorp"])
    
    data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.startswith("KF",na = False), 
                                         "AMAZON", data["DeviceCorp"])    
    
    for i in ['D', 'E', 'F', 'G']:
    
        data.loc[:, "DeviceCorp"] = np.where(data["DeviceCorp"].str.startswith(i,na = False), 
                                         "SONY", data["DeviceCorp"]) 
    
    data["DeviceCorp"] = data["DeviceCorp"].str.upper()
    
    less_200 = pd.DataFrame(data["DeviceCorp"].value_counts()).reset_index()
    less_200.columns = ["Device" , "DeviceCorp"]
    less_200 = less_200[less_200["DeviceCorp"] < 200]["Device"].tolist()
    
    data.loc[: , "DeviceCorp"] = np.where(data["DeviceCorp"].isin(less_200), 
                                          "OTHERS" , data["DeviceCorp"])
    
    
    
    return data


# In[161]:


train = device_transform(train)
test = device_transform(test)


# In[162]:


train["DeviceCorp"].unique()


# In[163]:


test["DeviceCorp"].unique()


# In[164]:


print("DevicCorp Nan Values Train : {}".format(train["DeviceCorp"].isnull().sum()))
print("DeviceCorp Nan Values Test  : {}".format(test["DeviceCorp"].isnull().sum()))


# In[165]:



device_corp = train.groupby("isFraud")["DeviceCorp"].value_counts(normalize = True).mul(100).              rename("percentage").reset_index().sort_values("percentage")

plt.figure(figsize = (15,15))

corp_bar = sns.barplot(x = "DeviceCorp", y = "percentage", data = device_corp, hue = "isFraud", 
                       palette = "viridis")

for p in corp_bar.patches:
        corp_bar.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2.,p.get_height()), 
                     ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')

size = 15
plt.title("Train DeviceCorp",fontsize = size)
plt.xlabel("DeviceCorp", fontsize = size)
plt.ylabel("Percentage",fontsize = size)
plt.xticks(rotation = 45, fontsize= size)
plt.show()


# ## Feature Enginerring

# In[166]:


train["Trans_min_mean"] = train["TransactionAmt"] - train["TransactionAmt"].mean()
test["Trans_min_mean"] = test["TransactionAmt"] - test["TransactionAmt"].mean()

train["Trans_min_std"] = train["Trans_min_mean"] / train["TransactionAmt"].std()
test["Trans_min_std"] = test["Trans_min_mean"] / test["TransactionAmt"].std()


train["TransactionAmt_to_mean_card1"]=                     train["TransactionAmt"]  / train.groupby(["card1"])["TransactionAmt"].transform("mean")

train["TransactionAmt_to_mean_card4"]=                     train["TransactionAmt"]  / train.groupby(["card4"])["TransactionAmt"].transform("mean")

train["TransactionAmt_to_std_card1"]=                     train["TransactionAmt"]  / train.groupby(["card1"])["TransactionAmt"].transform("std")

train["TransactionAmt_to_std_card4"]=                     train["TransactionAmt"]  / train.groupby(["card4"])["TransactionAmt"].transform("std")

test["TransactionAmt_to_mean_card1"]=                     test["TransactionAmt"]  / test.groupby(["card1"])["TransactionAmt"].transform("mean")

test["TransactionAmt_to_mean_card4"]=                     test["TransactionAmt"]  / test.groupby(["card4"])["TransactionAmt"].transform("mean")

test["TransactionAmt_to_std_card1"]=                     test["TransactionAmt"]  / test.groupby(["card1"])["TransactionAmt"].transform("std")

test["TransactionAmt_to_std_card4"]=                     test["TransactionAmt"]  / test.groupby(["card4"])["TransactionAmt"].transform("std")


# In[167]:


train.head()


# In[168]:


test.head()


# In[172]:


# Save Csv

train.to_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data/train.csv")
test.to_csv("/Users/gokhanersoz/Desktop/GitHub/Fraud/Fraud_Data/test.csv")

