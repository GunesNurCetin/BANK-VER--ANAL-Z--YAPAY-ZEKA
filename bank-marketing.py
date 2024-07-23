#   @GÜNEŞ NUR ÇETİN 

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from mixed_naive_bayes import MixedNB
from sklearn.metrics import classification_report


# VERI SETININ OKUNMASI
veriSeti = pd.read_csv("bank-full.csv", sep=";")

    
# VERIYI ANLAMA VE HAZIRLAMA
veriSeti.dtypes

label_encoder = LabelEncoder()

kategorikNitelikler = []
for nitelik in veriSeti.columns:
    if veriSeti[nitelik].dtype == "object":
        kategorikNitelikler.append(veriSeti.columns.get_loc(nitelik))
        veriSeti.loc[:,nitelik] = label_encoder.fit_transform(veriSeti.loc[:,nitelik])
        veriSeti[nitelik] = veriSeti[nitelik].astype("int64")

kategorikNitelikler.pop(len(kategorikNitelikler)-1)

veriSeti.dtypes


# Egitim ve test veri setinin olusturulmasi

# 5-kat capraz gecerleme icin egitim ve test indislerinin olusturulmasi
k = 5
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
for train_index, test_index in cv.split(X = veriSeti.iloc[:,0:16], y = veriSeti.y):
    print("Egitim Indisleri: ", train_index)
    print("Test Indisleri: ", test_index, "\n")

# Naive Bayes Siniflandirici modelinin cagirilmasi
dogruluk = []
F1 = []
k = 5
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)

for train_index, test_index in cv.split(X = veriSeti, y = veriSeti.y):    
    X_train, X_test, y_train, y_test = veriSeti.iloc[train_index,0:16], veriSeti.iloc[test_index, 0:16], veriSeti.iloc[train_index,16], veriSeti.iloc[test_index, 16]
    # Naive Bayes Siniflandiricisinin Olusturulmasi
    nb_model = MixedNB(categorical_features = kategorikNitelikler)
    nb_model.fit(X_train, y_train)
    # Naive Bayes Siniflandirici Tahminlerinin Elde Edilmesi
    y_tahmin = nb_model.predict(X_test)
    y_tahmin = label_encoder.inverse_transform(y_tahmin)
    y_test = label_encoder.inverse_transform(y_test)
    # Performans degerlendirme
    my_report = classification_report(y_true = y_test, y_pred = y_tahmin, labels=["no", "yes"], output_dict=True)
    dogruluk.append(my_report["accuracy"])
    F1.append(my_report["yes"]["f1-score"])


dogruluk
F1

# Modelin nihai performansi- dogruluk
np.mean(dogruluk).round(3)
# Modelin nihai performansi- COVID-19 F-olcusu
np.mean(F1).round(3)