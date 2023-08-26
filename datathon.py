#--------------------------------KUTUPHANELER--------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import config as cnf
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


#-----------------------------GORUNUM AYARLARI------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 700)

#------------------------SAMPLE SUBMISSION FONKSIYONU-------------------------
def sample_sub(y_pred):
  newlist = ["obek_" + str(item + 1) for item in y_pred]

  submission = pd.DataFrame({"id": range(0, 2340),
                           "Öbek İsmi": newlist})
  return submission

#------------------------DATA PREPROCESSING-------------------------
def data_prep(dataframe,outlier=True):
  #---------------------------DATAYI DUZENLEYELIM---------------------------
  utils.first_edit(dataframe)

  #---------------------INDEX DEGISKENINI DROP EDELIM-------------------------
  dataframe = dataframe.drop("INDEX",axis=1)

  #-------------------DEGISKENLERI TIPLERE GORE AYIRALIM----------------------
  cat_cols, num_cols = utils.grab_col_names(dataframe)

  #--------------------------OUTLIER INCELEMESI-------------------------------
  # Burada egitim datasi mi yoksa test datasi mi o ayriliyor
  if outlier:
    # Vahit Hocanın Fonksiyonu Kullanarak Abartı Bir Outlier Var Mi Bakalim
    for col in num_cols:
        print(f"{utils.check_outlier(dataframe,col)}   {col}")
    out_indices = utils.quantile_outlier(dataframe,num_cols,target = "OBEK_ISMI")

    # OVERFITTING
    # NIHAT OUTLIER FONKSIYON
    quantiel_indices = set(out_indices)
    quantiel_indicess = list(quantiel_indices)
    print(len(quantiel_indices))
    outdf = dataframe.loc[quantiel_indicess]
    print(type(outdf))
    outdf = pd.DataFrame(outdf)
    print(type(outdf))
    dataframe.drop(index=quantiel_indices, inplace=True)

    # Bir de LOF yontemiyle outlier incelemesi yapalim
    df = dataframe.select_dtypes(include=['float64', 'int64'])
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
    dataframe.drop(index=df[df_scores < th].index, inplace=True)

    # Nihat'in Ilk Fonksiyonu Ile Bakalim
    # Outlierlarin indeksleri bulundu
    for num_col in num_cols:
      indices = utils.diffrent_outlier(dataframe,num_col,"OBEK_ISMI")
    # Outlier degerler drop ediliyor
    # dataframe.drop(index=indices, inplace=True)

  #---------------------------FEATURE EXTRACTION-------------------------------
  # YENI_ORT_GELIR
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] < 14), "YENI_ORT_SEPET"] = 'CIMRI'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] >= 14) & (dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] <= 29), "YENI_ORT_SEPET"] = 'TUTUMLU'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] >= 29) & (dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] <= 50), "YENI_ORT_SEPET"] = 'NORMAL_INSAN'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] >= 50) & (dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] <= 100), "YENI_ORT_SEPET"] = 'VAR_KI_ALIYON'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] > 100), "YENI_ORT_SEPET"] = 'MUSRIF'
  
  
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_GELIR'] < 400000), "YENI_ORT_GELIR"] = 'EH_ISTE'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_GELIR'] >= 400000) & (dataframe['YILLIK_ORTALAMA_GELIR'] <= 600000), "YENI_ORT_GELIR"] = 'YASIYORSUN_HAYATI'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_GELIR'] > 600000), "YENI_ORT_GELIR"] = 'KOSEYI_DONMUS'

  dataframe["YENI_ALIM_GUCU"] = dataframe["YILLIK_ORTALAMA_GELIR"] / dataframe["YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI"]
  dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
  
  mean_value = dataframe["YENI_ALIM_GUCU"].mean()
  dataframe["YENI_ALIM_GUCU"].fillna(mean_value, inplace=True)
    
  for obek in dataframe["YENI_ORT_SEPET"].unique():
    index = dataframe[(dataframe["YENI_ORT_SEPET"] == obek) & (dataframe["YENI_ALIM_GUCU"].isna())].index
    mean_value = dataframe[dataframe["YENI_ORT_SEPET"] == obek]["YENI_ALIM_GUCU"].mean()
    dataframe.loc[index, "YENI_ALIM_GUCU"] = mean_value
    print(f"{obek} : {index}")

  utils.check_df(dataframe,non_numeric=False)

  # mean_value = dataframe["YENI_ALIM_GUCU"].mean()
  # dataframe["YENI_ALIM_GUCU"].fillna(mean_value, inplace=True)

  # OBEK ISMI ve YILLIK ORTALAMA SIPARIŞ VERILEN ÜRÜN ADEDI
  # print(dataframe.groupby("OBEK_ISMI")["YENI_ALIM_GUCU"].mean())
  # print("\n")
  
  
  
  # dataframe['YENI_BEKAR_ERKEK'] = (dataframe['MEDENI_DURUM'] == 'bekar') & (dataframe['CINSIYET'] == 'erkek')
  # dataframe['YENI_BEKAR_KADIN'] = (dataframe['MEDENI_DURUM'] == 'bekar') & (dataframe['CINSIYET'] == 'kadın')
  # dataframe['YENI_EVLI_ERKEK'] = (dataframe['MEDENI_DURUM'] == 'evli') & (dataframe['CINSIYET'] == 'erkek')
  # dataframe['YENI_EVLI_KADIN'] = (dataframe['MEDENI_DURUM'] == 'evli') & (dataframe['CINSIYET'] == 'kadın')

  # YENI_ALISVERIS_BAGIMLILIK_ORANI
  # dataframe["YENI_ALISVERIS_BAGIMLILIK_ORANI"] = dataframe["YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI"] / train_df["YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI"]

  # dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
  # mean_value = dataframe["YENI_ALISVERIS_BAGIMLILIK_ORANI"].mean()
  # dataframe["YENI_ALISVERIS_BAGIMLILIK_ORANI"].fillna(mean_value, inplace=True)




  # YENI_YUKSEK_FIYATLI_URUN_AVCISI  (XGBOOST BOZUYOR)
  # dataframe["YENI_YUKSEK_FIYATLI_URUN_AVCISI"] = dataframe["YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI"] / train_df["YILLIK_ORTALAMA_SATIN_ALIM_MIKTARI"]

  # YENI_ORT_SEPET


  cat_cols, num_cols = utils.grab_col_names(dataframe)
  return cat_cols, num_cols, dataframe

#---------------------------VERIYI ICE AKTARALIM----------------------------
train_df = pd.read_csv(cnf.train)

#----------------------------DATA PREPROCESSING-----------------------------
cat_cols, num_cols, df = data_prep(train_df)
dff = df
# utils.check_df(out_df,non_numeric=False)

#-----------------------MODEL ONCESI SON HAZIRLIK--------------------------
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


# ohe_cat_cols = [col for col in cat_cols if col in ["YENI_ORT_GELIR",]]
# new_cat_cols = [col for col in cat_cols if col not in ["CINSIYET","MEDENI_DURUM","EGITIME_DEVAM_ETME_DURUMU","YENI_ORT_GELIR"]]

ohe_cat_cols = [col for col in cat_cols if col in ["YENI_ORT_GELIR",]]
new_cat_cols = [col for col in cat_cols if col not in ["YENI_ORT_GELIR"]]


# df.drop(["CINSIYET","MEDENI_DURUM","EGITIME_DEVAM_ETME_DURUMU"], axis=1,inplace=True)
# df.drop(["EGITIME_DEVAM_ETME_DURUMU"], axis=1,inplace=True)

ohe_cat_cols = [col for col in cat_cols if col in ["YENI_ORT_GELIR"]]
new_cat_cols = [col for col in cat_cols if col not in ["EGITIME_DEVAM_ETME_DURUMU","YENI_ORT_GELIR","MEDENI_DURUM"]]
df.drop(["EGITIME_DEVAM_ETME_DURUMU","MEDENI_DURUM"], axis=1,inplace=True)
for col in new_cat_cols:
        df[col] = le.fit_transform(df[col])
df = pd.get_dummies(df, columns=ohe_cat_cols, drop_first=True)
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
# utils.hyperparameter_optimization(X,y,scoring="accuracy")



# TEST

test_df = pd.read_csv(cnf.test)
cat_cols, num_cols, test_dff = data_prep(test_df,outlier=False)
from sklearn.preprocessing import LabelEncoder
df_test = test_dff

le = LabelEncoder()

ohe_cat_cols = [col for col in cat_cols if col in ["YENI_ORT_GELIR"]]
new_cat_cols = [col for col in cat_cols if col not in ["EGITIME_DEVAM_ETME_DURUMU","YENI_ORT_GELIR","MEDENI_DURUM"]]

# ohe_cat_cols = [col for col in cat_cols if col in ["YENI_ORT_GELIR",]]
# new_cat_cols = [col for col in cat_cols if col not in ["EGITIME_DEVAM_ETME_DURUMU","YENI_ORT_GELIR"]]


df_test.drop(["EGITIME_DEVAM_ETME_DURUMU","MEDENI_DURUM"], axis=1,inplace=True)
# df.drop(["EGITIME_DEVAM_ETME_DURUMU"], axis=1,inplace=True)

# new_cat_cols = [col for col in cat_cols if col not in ["EGITIME_DEVAM_ETME_DURUMU"]]
# df.drop(["EGITIME_DEVAM_ETME_DURUMU"], axis=1,inplace=True)
for col in new_cat_cols:
        df_test[col] = le.fit_transform(df_test[col])
df_test = pd.get_dummies(df_test, columns=ohe_cat_cols, drop_first=True)

X_scaled_test = StandardScaler().fit_transform(df_test)

# scale ettikten sonra bir dizi döndürüyor ve bu dizide colum isimleri yok
# biz de aşağıdaki gibi yaparak onu düzeltiyoruz.
X_test = pd.DataFrame(X_scaled_test, columns=df_test.columns)

model = XGBClassifier(colsample_bytree = 0.5, learning_rate = 0.1, max_depth = 7, n_estimators = 75,num_class = [8], objective = ["multi:softmax"]).fit(X,y)

y_pred = model.predict(X_test)

sample_df = sample_sub(y_pred)

sample_df.to_csv("submission_9.csv",index=False)

y_test = pd.read_csv("submission_3.csv")

from sklearn.metrics import accuracy_score
# Gerçek ve tahmin edilen sınıfları birleştirme

# Accuracy hesaplaması
accuracy = accuracy_score(y_test["Öbek İsmi"], sample_df["Öbek İsmi"])

print("Accuracy:", accuracy)

# utils.importance(xgbm,X_test,y_pred,n_repeats=30,random_state=42)
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_pred, n_repeats=30, random_state=42)
importance_scores = result.importances_mean
sorted_indices = np.argsort(importance_scores)[::-1]

plt.barh(X_test.columns[sorted_indices], importance_scores[sorted_indices])
plt.xlabel('Permütasyon Önem Skorları')
plt.ylabel('Değişkenler')
plt.title('Değişkenlerin Permütasyon Önem Skorlari')
plt.show()