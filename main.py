import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 700)
pd.set_option('display.max_rows', None)

df = pd.read_csv("train.csv")

def chek_df(dataframe, head = 5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

chek_df(df)

# Numerik olan "Yıllık Ortalama Sepete Atılan Ürün Adedi" ve
# "Yıllık Ortalama Sipariş Verilen Ürün Adedi" değerlerinde
# outlier olma ihitmali yüksek. df'de Null değer yok.
# Z kuşağı falan olarak bakılabilir belki.
# öbek değişkeninin yaş ile alakası var gibi(özellikle 4,5,6,7,8)
# en zenginler öbek4 te
# fakirler 1 ve 2 ye ayrılmış
# aylık ortalama satın alım miktarı fazla olanlar öbek6
# aylık ortalama satın alım miktarı az olanlar öbek5 ve 2 ye

cat_cols = [col for col in df.columns if df[col].dtypes in ["object"]]
num_cols = [col for col in df.columns if df[col].dtypes not in ["object"] and col != "index"]

num_but_cat = [col for col in num_cols if df[col].nunique() < 10]
cat_but_car = [col for col in cat_cols if df[col].nunique() > 20]

df[cat_cols].nunique()

def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("############################################")

    if plot:
        sns.countplot(x = dataframe[col_name], y="Öbek İsmi", data = dataframe)
        plt.show(block = True)

for col in cat_cols:
    cat_summary(df, col, True)

def num_summary(dataframe, col_name, plot = True):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        dataframe[col_name].hist()
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block = True)

for col in num_cols:
    num_summary(df, col, True)

def target_summ_with_cat(dataframe, col_name):
    grouped = dataframe.groupby([col_name, "Öbek İsmi"]).size().reset_index(name="Kişi Sayısı")

    # Veriyi daha düzenli göstermek için pivot işlemi yapılıyor
    pivot_table = grouped.pivot(index=col_name, columns="Öbek İsmi", values="Kişi Sayısı").fillna(0)

    print(pivot_table)
    print("###########################################################")
    return pivot_table

for col in cat_cols:
    if col != "Öbek İsmi":
        target_summ_with_cat(df, col)

def target_summ_with_num(dataframe, col_name):
    print(dataframe.groupby("Öbek İsmi").agg({col_name: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summ_with_num(df, col)


a = target_summ_with_cat(df, "Yaş Grubu")

type(a)
a.columns
sns.countplot(a)
plt.show()

corr = df[num_cols].corr()

sns.set(rc = {'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap = "RdBu")
plt.show()

corr = df[num_cols].corr().abs()

df.groupby(["Eğitim Düzeyi", "Öbek İsmi"])["Yıllık Ortalama Sipariş Verilen Ürün Adedi"].mean()

# Veriyi daha düzenli göstermek için pivot işlemi yapılıyor
pivot_table = grouped.pivot(index="Eğitim Düzeyi", columns="Yıllık Ortalama Sipariş Verilen Ürün Adedi", values="Kişi Sayısı").fillna(0)

print(pivot_table)


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    grab_outliers(df, col)


low_limit, up_limit = outlier_thresholds(df, "Yıllık Ortalama Sepete Atılan Ürün Adedi")

check_outlier(df, "Yıllık Ortalama Sepete Atılan Ürün Adedi")


df.groupby("Öbek İsmi")["Yıllık Ortalama Sepete Atılan Ürün Adedi"].agg(["min", "max", "std", "mean"])

def scatter_figure(df, col_name, x, hue):
    plt.figure(figsize=(16, 11))
    sns.scatterplot(data=df, y=col_name, x=x, hue=hue)
    plt.show(block = True)

for col in num_cols:
    scatter_figure(df, col, "index", "Öbek İsmi")


[542, 2910, 4204, 4395, 4999, 5100, 5130]
df.drop(index=[542, 2910, 4204, 4395, 4999, 5100, 5130], inplace=True)
df.iloc[542]

from sklearn.neighbors import LocalOutlierFactor

df = df.drop("index",axis=1)
df = df.select_dtypes(include=['float64', 'int64'])


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]
#df_scores = -df_scores
np.sort(df_scores)[0:5]
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()
th = np.sort(df_scores)[7]
df[df_scores < th]
df[df_scores < th].shape
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
df[df_scores < th].index




df.iloc[542]

def scatter_figure(df, col_name, x, hue):
    plt.figure(figsize=(16, 11))
    sns.scatterplot(data=df, y=col_name, x=x, hue=hue)
    plt.show(block = True)

for col in num_cols:
    scatter_figure(df, col, "index", "Öbek İsmi")



plt.figure(figsize=(8, 6))  # Grafiğin boyutunu belirleyebilirsiniz
sns.countplot(data=df, x="Eğitim Düzeyi")  # countplot ile frekansları gösterelim
plt.xlabel("Kategoriler")
plt.ylabel("Frekans")
plt.title("Kategori Frekansları")
plt.xticks(rotation=45)  # Eğer kategori isimleri uzunsa x ekseninde rotasyon yapabilirsiniz
plt.show()


########################################################################################################

def outlier_new(df, col_name):

    dict = {}
    outlier_indices = []

    for obek in df["Öbek İsmi"].unique():
        selected_obek = df[df["Öbek İsmi"] == obek]
        selected_col = selected_obek[col_name]

        std = selected_col.std()
        avg = selected_col.mean()

        three_sigma_plus = avg + (3 * std)
        three_sigma_minus = avg - (3 * std)

        outlier_count = (selected_obek[col_name] > three_sigma_plus).sum() + (selected_obek[col_name]< three_sigma_minus).sum()
        dict.update({obek: outlier_count})

        outliers = selected_obek[(selected_col > three_sigma_plus) | (selected_col < three_sigma_minus)]
        outlier_indices.extend(outliers.index.tolist())

    print(col_name)
    print(dict)
    print(sum(list(dict.values())))
    print("##############################")
    return outlier_indices

indices = []

for col in num_cols:
    indices.extend(outlier_new(df, col))

indices = set(indices)

len(indices)

df.drop(index= indices, inplace=True)

def scatter_figure(df, col_name, x, hue):
    plt.figure(figsize=(16, 11))
    sns.scatterplot(data=df, y=col_name, x=x, hue=hue)
    plt.show(block = True)

for col in num_cols:
    scatter_figure(df, col, "index", "Öbek İsmi")

df.drop(["index", "Cinsiyet", "Medeni Durum", "Eğitime Devam Etme Durumu"], axis=1, inplace=True)

df.to_csv('data_outlier_cikmis_hali.csv', index=False)


################################################################################################


from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data_outlier_cikmis_hali.csv")

le = LabelEncoder()

new_cat_cols = [col for col in cat_cols if col not in ["Cinsiyet", "Medeni Durum", "Eğitime Devam Etme Durumu"]]

for col in new_cat_cols:
        df[col] = le.fit_transform(df[col])

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

y = df["Öbek İsmi"]
X = df.drop(["Öbek İsmi"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

# scale ettikten sonra bir dizi döndürüyor ve bu dizide colum isimleri yok
# biz de aşağıdaki gibi yaparak onu düzeltiyoruz.
X = pd.DataFrame(X_scaled, columns=X.columns)

################################################
# 3. Modeling & Prediction
################################################

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)

random_user["Öbek İsmi"]
knn_model.predict(random_user)

df.iloc[1160]

y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))