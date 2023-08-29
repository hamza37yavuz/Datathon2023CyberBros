import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import datathon
import config as cnf
import utils

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
from sklearn.ensemble import RandomForestClassifier
def data_prep(dataframe,test=True):
  #---------------------------DATAYI DUZENLEYELIM---------------------------
  utils.first_edit(dataframe)

  #---------------------INDEX DEGISKENINI DROP EDELIM-------------------------
  dataframe.drop("INDEX",axis=1,inplace = True)

  #-------------------DEGISKENLERI TIPLERE GORE AYIRALIM----------------------
  cat_cols, num_cols = utils.grab_col_names(dataframe)

  #--------------------------OUTLIER INCELEMESI-------------------------------
  # Burada egitim datasi mi yoksa test datasi mi o ayriliyor
  if test==False:
    # Vahit Hocanın Fonksiyonu Kullanarak Abartı Bir Outlier Var Mi Bakalim
    # for col in num_cols:
    #     print(f"{utils.check_outlier(dataframe,col)}   {col}")
    out_indices = utils.quantile_outlier(dataframe,num_cols,target = "OBEK_ISMI")

    # OVERFITTING
    # NIHAT OUTLIER FONKSIYON
    quantiel_indices = set(out_indices)
    quantiel_indicess = list(quantiel_indices)
    # print(len(quantiel_indices))
    outdf = dataframe.loc[quantiel_indicess]
    # print(type(outdf))
    outdf = pd.DataFrame(outdf)
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
    dataframe.drop(index=indices, inplace=True)
    print(len(quantiel_indices))
    print(len(indices))
    print(len(df[df_scores < th].index))
    

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
  
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] < 14), "YENI_KATEGORIK_DEGISKEN"] = 'CIMRI'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] >= 14) & (dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] <= 29), "YENI_KATEGORIK_DEGISKEN"] = 'TUTUMLU'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] >= 29) & (dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] <= 50), "YENI_KATEGORIK_DEGISKEN"] = 'NORMAL_INSAN'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] >= 50) & (dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] <= 100), "YENI_KATEGORIK_DEGISKEN"] = 'VAR_KI_ALIYON'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI'] > 100), "YENI_KATEGORIK_DEGISKEN"] = 'MUSRIF'

  dataframe.loc[(dataframe['YILLIK_ORTALAMA_GELIR'] < 400000), "YENI_KATEGORIK_DEGISKEN"] += '_EH_ISTE'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_GELIR'] >= 400000) & (dataframe['YILLIK_ORTALAMA_GELIR'] <= 600000), "YENI_KATEGORIK_DEGISKEN"] += '_YASIYORSUN_HAYATI'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_GELIR'] > 600000), "YENI_KATEGORIK_DEGISKEN"] += '_KOSEYI_DONMUS'

  dataframe["YENI_SATIN_AL_GEL"] = dataframe["YILLIK_ORTALAMA_SATIN_ALIM_MIKTARI"] / dataframe["YILLIK_ORTALAMA_GELIR"]
  
  dataframe["YENI_ORT_GEL_SEP"] = dataframe["YILLIK_ORTALAMA_GELIR"] / dataframe["YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI"]
  
  dataframe["YENI_ORT_SIP_SEP"] = dataframe["YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI"] / dataframe["YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI"]
  
  dataframe["YENI_ALIM_GUCU"] = dataframe["YILLIK_ORTALAMA_GELIR"] / dataframe["YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI"]
  dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    
  for obek in dataframe["YENI_ORT_SEPET"].unique():
    index = dataframe[(dataframe["YENI_ORT_SEPET"] == obek) & (dataframe["YENI_ALIM_GUCU"].isna())].index
    mean_value = dataframe[dataframe["YENI_ORT_SEPET"] == obek]["YENI_ALIM_GUCU"].mean()
    dataframe.loc[index, "YENI_ALIM_GUCU"] = mean_value
    # print(f"{obek} : {index}")
      
  # ÖBEK İSMI ve YILLIK ORTALAMA SIPARIŞ VERILEN ÜRÜN ADEDI
  # print(dataframe.groupby("OBEK_ISMI")["YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI"].median())
  # print("\n")
  # utils.check_df(dataframe,non_numeric=False)
  
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI'] < 8.5), "YENI_SIP_CAT"] = 'A'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI'] >= 8.5) & (dataframe['YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI'] <= 12), "YENI_SIP_CAT"] = 'B'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI'] >= 12) & (dataframe['YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI'] <= 16.5), "YENI_SIP_CAT"] = 'C'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI'] >= 16.5) & (dataframe['YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI'] <= 24.75), "YENI_SIP_CAT"] = 'D'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI'] >= 24.75) & (dataframe['YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI'] <= 34), "YENI_SIP_CAT"] = 'E'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI'] >= 34) & (dataframe['YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI'] <= 44.25), "YENI_SIP_CAT"] = 'F'
  dataframe.loc[(dataframe['YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI'] >= 44.25), "YENI_SIP_CAT"] = 'STOKCU'
  

  cat_cols, num_cols = utils.grab_col_names(dataframe)

  
  #-----------------------MODEL ONCESI SON HAZIRLIK--------------------------
  cat_cols = [col for col in cat_cols if col not in ["EGITIME_DEVAM_ETME_DURUMU"]]
  dataframe.drop(["EGITIME_DEVAM_ETME_DURUMU"], axis=1,inplace=True)
  
  if test:
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=True)
  else:
    cat_cols = [col for col in cat_cols if col not in ["OBEK_ISMI"]]
    le = LabelEncoder()
    dataframe['OBEK_ISMI'] = le.fit_transform(dataframe['OBEK_ISMI'])
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=True)
  print(cat_cols)
  return cat_cols, num_cols, dataframe



train = pd.read_csv("train.csv")
test = pd.read_csv("test_x.csv")

cat_cols, num_cols_train, train = datathon.data_prep(train,test = False)
cat_cols, num_cols_test, test = datathon.data_prep(test)

train["type"]="train"
test["type"]="test"
df= pd.concat([train,test])

for i in range(len(num_cols_train)):
    j = i+1
    for j in range(len(num_cols_test)):
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        sns.scatterplot(data=df, x=num_cols_train[i], y=num_cols_test[i+1],hue="type")
        plt.show(block = True)
    


# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.scatterplot(data=df, x="YILLIK_ORTALAMA_GELIR", y="YILLIK_ORTALAMA_SATIN_ALIM_MIKTARI",hue="type")
# plt.show(block = True)




# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.scatterplot(data=df, x="YILLIK_ORTALAMA_GELIR", y="YILLIK_ORTALAMA_SATIN_ALIM_MIKTARI",hue="type")
# plt.show(block = True)


# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.scatterplot(data=df, x="YILLIK_ORTALAMA_GELIR", y="YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI",hue="type")
# plt.show(block = True)

# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.scatterplot(data=df, x="YILLIK_ORTALAMA_GELIR", y="YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI",hue="type")
# plt.show(block = True)

# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.scatterplot(data=df, x="YILLIK_ORTALAMA_SATIN_ALIM_MIKTARI", y="YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI",hue="type")
# plt.show(block = True)

# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.scatterplot(data=df, x="YILLIK_ORTALAMA_SATIN_ALIM_MIKTARI", y="YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI",hue="type")
# plt.show(block = True)

# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.scatterplot(data=df, x="YILLIK_ORTALAMA_SATIN_ALIM_MIKTARI", y="YENI_ALIM_GUCU",hue="type")
# plt.show(block = True)

# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.scatterplot(data=df, x="YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI", y="YENI_ALIM_GUCU",hue="type")
# plt.show(block = True)

# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.scatterplot(data=df, x="YILLIK_ORTALAMA_SIPARIS_VERILEN_URUN_ADEDI", y="YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI",hue="type")
# plt.show(block = True)

# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.scatterplot(data=df, x="YILLIK_ORTALAMA_SEPETE_ATILAN_URUN_ADEDI", y="YENI_ALIM_GUCU",hue="type")
# plt.show(block = True)



