import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import datathon
import config as cnf
import utils

train = pd.read_csv("train.csv")
test = pd.read_csv("test_x.csv")

cat_cols, num_cols,train = datathon.data_prep(train,outlier=False)
cat_cols, num_cols,test = datathon.data_prep(test,outlier=False)

train["type"]="train"
test["type"]="test"
df= pd.concat([train,test])

# for i in range(num_cols):
#     sns.set(rc={'figure.figsize':(11.7,8.27)})
#     sns.scatterplot(data=df, x=num_cols[i], y=num_cols[i],hue="type")
#     plt.show(block = True)
    


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



