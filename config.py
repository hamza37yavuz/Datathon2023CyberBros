from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np


train = "train.csv"
test = "test_x.csv"

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [10, 11, None],
             "max_features": [7, 8, "sqrt"],
             "min_samples_split": [17,18,19],
             "n_estimators": [220,230,240]}
import warnings

# Tüm uyarıları geçici olarak filtrelemek
warnings.filterwarnings("ignore")
# rf_params = {
#     'n_estimators': np.arange(10, 200, 10),
#     'max_depth': [None] + list(np.arange(5, 30, 5)),
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'min_samples_split': np.arange(2, 11),
#     'min_samples_leaf': np.arange(1, 11),
#     'bootstrap': [True, False]
# }

# xgboost_params = {"learning_rate": [0.1, 0.05],
#                   "max_depth": [5,6,7],
#                   "n_estimators": [ 55,60,65],
#                   "colsample_bytree": [0.5, 1]}
xgboost_params = {"learning_rate": [0.1],
                  "max_depth": [4,5,6],
                  "n_estimators": [50,60,70],
                  "colsample_bytree": [0.5],
                  "objective": ['multi:softmax'],
                  "num_class":[8]}

lightgbm_params = {"learning_rate": [0.01,0.1],
                   "n_estimators": [400,500],
                   "colsample_bytree": [0.7, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(max_features='sqrt'), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(verbose = -1), lightgbm_params)]

# classifiers = [('LightGBM', LGBMClassifier(), lightgbm_params)]
# classifiers = [('KNN', KNeighborsClassifier(), knn_params),
#                ("CART", DecisionTreeClassifier(), cart_params),
#                ("RF", RandomForestClassifier(max_features='sqrt'), rf_params),
#                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params)]

classifiers = [('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params)]

# classifiers = [('LightGBM', LGBMClassifier(verbose = -1), lightgbm_params)]
# classifiers = [("RF", RandomForestClassifier(max_features='sqrt'), rf_params)]
# ("RF", RandomForestClassifier(max_features='sqrt'), rf_params)

"""
knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)


knn_model.predict(random_user)

df.iloc[1160]

y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))


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

print(cv_results['test_accuracy'].mean())
# 0.9738259727784564
print(cv_results['test_f1_macro'].mean())
# 0.9724793683420145
"""

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