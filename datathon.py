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
from sklearn.preprocessing import LabelEncoder

#-----------------------------GORUNUM AYARLARI------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 700)

#---------------------------VERIYI ICE AKTARALIM----------------------------
train_df = pd.read_csv("train.csv")

#---------------------COLUMN ISIMLERINI BUYUK HARF YAPALIM------------------
train_df.columns = [col.upper() for col in train_df.columns]
train_df.columns = train_df.columns.str.replace(' ', '_')
train_df.columns = train_df.columns.str.replace('İ', 'I')
train_df.columns = train_df.columns.str.replace('Ğ', 'G')
train_df.columns = train_df.columns.str.replace('Ö', 'O')
train_df.columns = train_df.columns.str.replace('Ü', 'U')

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
print(train_df.groupby("OBEK_ISMI")["YILLIK_ORTALAMA_SIPARIŞ_VERILEN_URUN_ADEDI"].mean())
print("\n")
# ÖBEK İSMI ve YILLIK ORTALAMA SEPETE ATILAN ÜRÜN ADEDI
print(train_df.groupby("OBEK_ISMI")["YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI"].mean())
print("\n")
# ÖBEK İSMI ve YILLIK ORTALAMA SATIN ALIM MIKTARI
print(train_df.groupby("OBEK_ISMI")["YILLIK_ORTALAMA_SATIN_ALIM_MIKTARI"].sum())
print("\n")
# ÖBEK İSMI ve YILLIK ORTALAMA GELIR
print(train_df.groupby("OBEK_ISMI")["YILLIK_ORTALAMA_GELIR"].median())
print("\n")
# ÖBEK İSMI ve YILLIK ORTALAMA GELIR
print(train_df.groupby(["OBEK_ISMI","MEDENI_DURUM"])["YILLIK_ORTALAMA_GELIR"].median())
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

indices = utils.outlier_new(train_df,num_cols)


#---------------------------KNN MODELINI KURALIM-------------------------------
from sklearn.preprocessing import LabelEncoder

df = train_df

le = LabelEncoder()

# new_cat_cols = [col for col in cat_cols if col not in ["CINSIYET", "MEDENI_DURUM", "EGITIME_DEVAM_ETME_DURUMU"]]
# df.drop(["CINSIYET", "MEDENI_DURUM", "EGITIME_DEVAM_ETME_DURUMU"], axis=1,inplace=True)
new_cat_cols = [col for col in cat_cols if col not in ["EGITIME_DEVAM_ETME_DURUMU"]]
df.drop(["EGITIME_DEVAM_ETME_DURUMU"], axis=1,inplace=True)
for col in new_cat_cols:
        df[col] = le.fit_transform(df[col])

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate

y = df["OBEK_ISMI"]
X = df.drop(["OBEK_ISMI"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

# scale ettikten sonra bir dizi döndürüyor ve bu dizide colum isimleri yok
# biz de aşağıdaki gibi yaparak onu düzeltiyoruz.
X = pd.DataFrame(X_scaled, columns=X.columns)

################################################
# 3. Modeling & Prediction
################################################

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)


knn_model.predict(random_user)

df.iloc[1160]

y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
'''
              precision    recall  f1-score   support
           0       0.96      0.97      0.96       668
           1       0.97      0.95      0.96       523
           2       0.95      0.95      0.95       668
           3       1.00      0.99      0.99       680
           4       0.99      1.00      1.00       681
           5       1.00      1.00      1.00       664
           6       0.99      1.00      1.00       660
           7       1.00      1.00      1.00       690
    accuracy                           0.98      5234
   macro avg       0.98      0.98      0.98      5234
weighted avg       0.98      0.98      0.98      5234
'''


cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1_macro"])

cv_results['test_accuracy'].mean()
# 0.9738259727784564
cv_results['test_f1_macro'].mean()
# 0.9724793683420145

############################################################################################3

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

# {'n_neighbors': 5}
knn_gs_best.best_params_

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1_macro"])

cv_results['test_accuracy'].mean()
# 0.9738259727784564
cv_results['test_f1_macro'].mean()
# 0.9724793683420145

############################################################################################3

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

# {'n_neighbors': 5}