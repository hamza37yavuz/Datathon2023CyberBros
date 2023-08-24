from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



train = "train.csv"
test = "test_x.csv"

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [18,19,20],
             "n_estimators": [240,300]}

xgboost_params = {"learning_rate": [0.1, 0.05],
                  "max_depth": [7,8],
                  "n_estimators": [ 60, 80],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

# classifiers = [('KNN', KNeighborsClassifier(), knn_params),
#                ("CART", DecisionTreeClassifier(), cart_params),
#                ("RF", RandomForestClassifier(max_features='sqrt'), rf_params),
#                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
#                ('LightGBM', LGBMClassifier(), lightgbm_params)]

# classifiers = [('LightGBM', LGBMClassifier(), lightgbm_params)]
# classifiers = [('KNN', KNeighborsClassifier(), knn_params),
#                ("CART", DecisionTreeClassifier(), cart_params),
#                ("RF", RandomForestClassifier(max_features='sqrt'), rf_params),
#                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params)]

classifiers = [('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params)]
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
