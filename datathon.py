#--------------------------------KUTUPHANELER--------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import config as cnfg
import utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier


#-----------------------------GORUNUM AYARLARI------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 700)

#---------------------------VERIYI ICE AKTARALIM----------------------------
train_df = pd.read_csv("train.csv")

#---------------------COLUMN ISIMLERINI BUYUK HARF YAPALIM------------------
train_df.columns = [col.upper() for col in train_df.columns]
train_df.columns = train_df.columns.str.replace(' ', '_')

#---------------------INDEX DEGISKENINI DROP EDELIM-------------------------
train_df = train_df.drop("INDEX",axis=1)

#--------------------------DATAYI ANALIZ EDELIM-----------------------------
utils.check_df(train_df,non_numeric=False)

#-------------------DEGISKENLERI TIPLERE GORE AYIRALIM----------------------
cat_cols, num_cols, cat_but_car = utils.grab_col_names(train_df)

#--------------------------KORELASYON MATRISI-------------------------------
# utils.correlation_matrix(train_df,num_cols)

#--------------------------OUTLIER INCELEMESI-------------------------------
for col in num_cols:
    print(f"{utils.check_outlier(train_df,col)}   {col}")
# Yukarida outlier bulamadik ama assa
from sklearn.neighbors import LocalOutlierFactor

df = train_df.select_dtypes(include=['float64', 'int64'])

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# df_scores = -df_scores
np.sort(df_scores)[0:5]
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
# plt.show()
th = np.sort(df_scores)[3]
df[df_scores < th]
df[df_scores < th].shape
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
df[df_scores < th].index
train_df.drop(index=df[df_scores < th].index, inplace=True)

    
#----------------NUMERIK VE KATEGORIK DEGISKEN INCELEMESI-------------------
for num_col in num_cols:
  utils.num_summary(train_df,num_col,plot=False)
for cat_col in cat_cols:
  utils.cat_summary(train_df,cat_col,plot=False)
  
#-------------------------------GROUPBY-------------------------------------
# ÖBEK İSMI ve YILLIK ORTALAMA SIPARIŞ VERILEN ÜRÜN ADEDI
print(train_df.groupby("ÖBEK_İSMI")["YILLIK_ORTALAMA_SIPARIŞ_VERILEN_ÜRÜN_ADEDI"].mean())
print("\n")
# ÖBEK İSMI ve YILLIK ORTALAMA SEPETE ATILAN ÜRÜN ADEDI
print(train_df.groupby("ÖBEK_İSMI")["YILLIK_ORTALAMA_SEPETE_ATILAN_ÜRÜN_ADEDI"].mean())
print("\n")
# ÖBEK İSMI ve YILLIK ORTALAMA SATIN ALIM MIKTARI
print(train_df.groupby("ÖBEK_İSMI")["YILLIK_ORTALAMA_SATIN_ALIM_MIKTARI"].sum())
print("\n")
# ÖBEK İSMI ve YILLIK ORTALAMA GELIR
print(train_df.groupby("ÖBEK_İSMI")["YILLIK_ORTALAMA_GELIR"].median())
print("\n")
# ÖBEK İSMI ve YILLIK ORTALAMA GELIR
print(train_df.groupby(["ÖBEK_İSMI","MEDENI_DURUM"])["YILLIK_ORTALAMA_GELIR"].median())
print("\n")

#---------------------------FEATURE EXTRACTION-------------------------------
train_df.loc[(train_df['YILLIK_ORTALAMA_GELIR'] < 400000), "YENI_ORT_GELIR"] = 'EH_ISTE'
train_df.loc[(train_df['YILLIK_ORTALAMA_GELIR'] >= 400000) & (train_df['YILLIK_ORTALAMA_GELIR'] <= 600000), "YENI_ORT_GELIR"] = 'YASIYORSUN_HAYATI'
train_df.loc[(train_df['YILLIK_ORTALAMA_GELIR'] > 600000), "YENI_ORT_GELIR"] = 'KOSEYI_DONMUS'

# train_df.loc[(train_df['YILLIK ORTALAMA SEPETE ATILAN ÜRÜN ADEDI'] <9) & (train_df['MEDENI DURUM'] == "bekar"), "YENI_MUSTERI_PROFIL"] = 'BEKAR_CIMRI'
# train_df.loc[(train_df['YILLIK ORTALAMA SEPETE ATILAN ÜRÜN ADEDI'] <13) & (train_df['MEDENI DURUM'] == "evli"), "YENI_MUSTERI_PROFIL"] = 'EVLI_CIMRI'
# train_df.loc[(train_df['YILLIK ORTALAMA SEPETE ATILAN ÜRÜN ADEDI'] >= 400000) & (train_df['YILLIK ORTALAMA GELIR'] <= 600000), "YENI_ORT_GELIR"] = 'YASIYORSUN_HAYATI'
# train_df.loc[(train_df['YILLIK ORTALAMA SEPETE ATILAN ÜRÜN ADEDI'] > 600000), "YENI_ORT_GELIR"] = 'KOSEYI_DONMUS'
cat_cols, num_cols, cat_but_car = utils.grab_col_names(train_df)
utils.check_df(train_df,non_numeric=False)
#---------------------------KNN MODELINI KURALIM-------------------------------
df = train_df.drop(["ÖBEK_İSMI"], axis=1)
train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)

y = train_df["ÖBEK_İSMI"]
X = train_df.drop(["ÖBEK_İSMI"], axis=1)

knn = KNeighborsClassifier(n_neighbors=5,metric = "minkowski")
knn.fit(X,y)

