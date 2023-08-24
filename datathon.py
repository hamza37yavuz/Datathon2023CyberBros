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
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier

#------------------------SAMPLE SUBMISSION FONKSIYONU-------------------------
def sample_sub(y_pred):
  newlist = ["obek_" + str(item + 1) for item in y_pred]

  submission = pd.DataFrame({"id": range(0, 2340),
                           "Öbek İsmi": newlist})
  return submission


#-----------------------------GORUNUM AYARLARI------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 700)

#---------------------------VERIYI ICE AKTARALIM----------------------------
train_df = pd.read_csv("train.csv")

def data_prep(train_df,outlier=True):
  #---------------------COLUMN ISIMLERINI BUYUK HARF YAPALIM------------------
  train_df.columns = [col.upper() for col in train_df.columns]
  train_df.columns = train_df.columns.str.replace(' ', '_')
  train_df.columns = train_df.columns.str.replace('İ', 'I')
  train_df.columns = train_df.columns.str.replace('Ğ', 'G')
  train_df.columns = train_df.columns.str.replace('Ö', 'O')
  train_df.columns = train_df.columns.str.replace('Ü', 'U')
  train_df.columns = train_df.columns.str.replace('Ş', 'S')

  #---------------------INDEX DEGISKENINI DROP EDELIM-------------------------
  train_df = train_df.drop("INDEX",axis=1)

  #--------------------------DATAYI ANALIZ EDELIM-----------------------------
  # utils.check_df(train_df,non_numeric=False)

  #-------------------DEGISKENLERI TIPLERE GORE AYIRALIM----------------------
  cat_cols, num_cols, cat_but_car = utils.grab_col_names(train_df)
  #--------------------------KORELASYON MATRISI-------------------------------
  # utils.correlation_matrix(train_df,num_cols)
  if outlier:
    #--------------------------OUTLIER INCELEMESI-------------------------------
    # Utils dosyasindaki fonksiyonu kullanarak abartı bir outlier var mı bakalim
    for col in num_cols:
        print(f"{utils.check_outlier(train_df,col)}   {col}")
    out_indices = utils.quantile_outlier(train_df,num_cols,target = "OBEK_ISMI")

    # NIHAT OUTLIER FONKSIYON
    quantiel_indices = set(out_indices)
    print(len(quantiel_indices))
    train_df.drop(index=quantiel_indices, inplace=True)
        
    # Bir de lof yontemiyle outlier incelemesi yapalim
    df = train_df.select_dtypes(include=['float64', 'int64'])
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(df)
    df_scores = clf.negative_outlier_factor_
    df_scores[0:5]
    # df_scores = -df_scores
    np.sort(df_scores)[0:5]
    
    # Gorsellestirme
    # scores = pd.DataFrame(np.sort(df_scores))
    # scores.plot(stacked=True, xlim=[0, 50], style='.-')
    # plt.show()
    
    th = np.sort(df_scores)[3]
    df[df_scores < th]
    df[df_scores < th].shape
    # df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

    # Outlierlarin indeksleri bulundu
    df[df_scores < th].index
    #Outlier degerler drop ediliyor
    train_df.drop(index=df[df_scores < th].index, inplace=True)

    # Outlierlarin indeksleri bulundu
    for num_col in num_cols:
      indices = utils.diffrent_outlier(train_df,num_col,"OBEK_ISMI")
    # Outlier degerler drop ediliyor
    train_df.drop(index=indices, inplace=True)

  #---------------------------FEATURE EXTRACTION-------------------------------
  train_df.loc[(train_df['YILLIK_ORTALAMA_GELIR'] < 400000), "YENI_ORT_GELIR"] = 'EH_ISTE'
  train_df.loc[(train_df['YILLIK_ORTALAMA_GELIR'] >= 400000) & (train_df['YILLIK_ORTALAMA_GELIR'] <= 600000), "YENI_ORT_GELIR"] = 'YASIYORSUN_HAYATI'
  train_df.loc[(train_df['YILLIK_ORTALAMA_GELIR'] > 600000), "YENI_ORT_GELIR"] = 'KOSEYI_DONMUS'

  
  
  # train_df["YENI_ALISVERIS_BAGIMLILIK_ORANI"] = train_df["YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI"] / train_df["YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI"]

  # XGBOOST BOZUYOR
  # train_df["YENI_YUKSEK_FIYATLI_URUN_AVCISI"] = train_df["YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI"] / train_df["YILLIK_ORTALAMA_SATIN_ALIM_MIKTARI"]
  
  train_df.loc[(train_df['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] < 14), "YENI_ORT_SEPET"] = 'CIMRI'
  train_df.loc[(train_df['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] >= 14) & (train_df['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] <= 29), "YENI_ORT_SEPET"] = 'TUTUMLU'
  train_df.loc[(train_df['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] >= 29) & (train_df['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] <= 50), "YENI_ORT_SEPET"] = 'NORMAL_INSAN'
  train_df.loc[(train_df['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] >= 50) & (train_df['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] <= 100), "YENI_ORT_SEPET"] = 'VAR_KI_ALIYON'
  train_df.loc[(train_df['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] > 100), "YENI_ORT_SEPET"] = 'MUSRIF'
  
  cat_cols, num_cols, cat_but_car = utils.grab_col_names(train_df)
  return cat_cols, num_cols, train_df

cat_cols, num_cols, df = data_prep(train_df)

#----------------NUMERIK VE KATEGORIK DEGISKEN INCELEMESI-------------------
# for num_col in num_cols:
#   utils.num_summary(train_df,num_col,plot=False)
# for cat_col in cat_cols:
#   utils.cat_summary(train_df,cat_col,plot=False)
  
#-------------------------------GROUPBY-------------------------------------
# ÖBEK İSMI ve YILLIK ORTALAMA SIPARIŞ VERILEN ÜRÜN ADEDI
# print(train_df.groupby("OBEK_ISMI")["YILLIK_ORTALAMA_SIPARIŞ_VERILEN_URUN_ADEDI"].mean())
# print("\n")
# # ÖBEK İSMI ve YILLIK ORTALAMA SEPETE ATILAN ÜRÜN ADEDI
# print(train_df.groupby("OBEK_ISMI")["YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI"].mean())
# print("\n")
# # ÖBEK İSMI ve YILLIK ORTALAMA SATIN ALIM MIKTARI
# print(train_df.groupby("OBEK_ISMI")["YILLIK_ORTALAMA_SATIN_ALIM_MIKTARI"].sum())
# print("\n")
# # ÖBEK İSMI ve YILLIK ORTALAMA GELIR
# print(train_df.groupby("OBEK_ISMI")["YILLIK_ORTALAMA_GELIR"].median())
# print("\n")
# # ÖBEK İSMI ve YILLIK ORTALAMA GELIR
# print(train_df.groupby(["OBEK_ISMI","MEDENI_DURUM"])["YILLIK_ORTALAMA_GELIR"].median())
# print("\n")


#-----------------------MODEL ONCESI SON HAZIRLIK--------------------------
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()

# new_cat_cols = [col for col in cat_cols if col not in ["CINSIYET", "MEDENI_DURUM", "EGITIME_DEVAM_ETME_DURUMU"]]
# df.drop(["CINSIYET", "MEDENI_DURUM", "EGITIME_DEVAM_ETME_DURUMU"], axis=1,inplace=True)
new_cat_cols = [col for col in cat_cols if col not in ["EGITIME_DEVAM_ETME_DURUMU"]]
df.drop(["EGITIME_DEVAM_ETME_DURUMU"], axis=1,inplace=True)
for col in new_cat_cols:
        df[col] = le.fit_transform(df[col])

#-----------------------MODEL--------------------------

y = df["OBEK_ISMI"]
X = df.drop(["OBEK_ISMI"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

# scale ettikten sonra bir dizi döndürüyor ve bu dizide colum isimleri yok
# biz de aşağıdaki gibi yaparak onu düzeltiyoruz.
X = pd.DataFrame(X_scaled, columns=X.columns)


import warnings

# Tüm uyarıları geçici olarak filtrelemek
warnings.filterwarnings("ignore")
utils.hyperparameter_optimization(X,y,scoring="accuracy")



# TEST

# test_df = pd.read_csv("test_x.csv")
# cat_cols, num_cols, test_dff = data_prep(test_df,outlier=False)
# from sklearn.preprocessing import LabelEncoder
# df_test = test_dff

# le = LabelEncoder()

# # new_cat_cols = [col for col in cat_cols if col not in ["CINSIYET", "MEDENI_DURUM", "EGITIME_DEVAM_ETME_DURUMU"]]
# # df.drop(["CINSIYET", "MEDENI_DURUM", "EGITIME_DEVAM_ETME_DURUMU"], axis=1,inplace=True)

# new_cat_cols = [col for col in cat_cols if col not in ["EGITIME_DEVAM_ETME_DURUMU"]]
# df_test.drop(["EGITIME_DEVAM_ETME_DURUMU"], axis=1,inplace=True)

# for col in new_cat_cols:
#         df_test[col] = le.fit_transform(df_test[col])

# X_scaled_test = StandardScaler().fit_transform(df_test)

# # scale ettikten sonra bir dizi döndürüyor ve bu dizide colum isimleri yok
# # biz de aşağıdaki gibi yaparak onu düzeltiyoruz.
# X_test = pd.DataFrame(X_scaled_test, columns=df_test.columns) 
        
# model = XGBClassifier(colsample_bytree = 0.5, learning_rate = 0.1, max_depth = 7, n_estimators = 60).fit(X,y)

# y_pred = model.predict(X_test)

# sample_df = sample_sub(y_pred)

# sample_df.to_csv("submission_5.csv",index=False)


# utils.importance(xgbm,X_test,y_pred,n_repeats=30,random_state=42)
# from sklearn.inspection import permutation_importance

# result = permutation_importance(xgbm, X_test, y_pred, n_repeats=30, random_state=42)
# importance_scores = result.importances_mean
# sorted_indices = np.argsort(importance_scores)[::-1]

# plt.barh(X_test.columns[sorted_indices], importance_scores[sorted_indices])
# plt.xlabel('Permütasyon Önem Skorları')
# plt.ylabel('Değişkenler')
# plt.title('Değişkenlerin Permütasyon Önem Skorları')
# plt.show()