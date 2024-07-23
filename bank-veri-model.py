#   @GÜNEŞ NUR ÇETİN 
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mixed_naive_bayes import MixedNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# VERI SETININ OKUNMASI
veriSeti = pd.read_csv("bank-full.csv", sep=";")

veriSeti.describe()

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
X_train, X_test, y_train, y_test = train_test_split(veriSeti.iloc[:,0:16], veriSeti.iloc[:,16], test_size=0.3, random_state=1)



# MODELLEME
# Naive Bayes Siniflandiricisinin Olusturulmasi
nb_model = MixedNB(categorical_features = kategorikNitelikler)
nb_model.fit(X_train, y_train)

# Naive Bayes Siniflandirici Tahminlerinin Elde Edilmesi
y_tahmin = nb_model.predict(X_test)
y_tahmin = label_encoder.inverse_transform(y_tahmin)
y_test = label_encoder.inverse_transform(y_test)


# PERFORMAN DEGERLENDIRME
my_cm = confusion_matrix(y_true = y_test, y_pred = y_tahmin, labels=["no","yes"])

my_cm_p = ConfusionMatrixDisplay(my_cm, display_labels=["no", "yes"])
my_cm_p.plot()

tn, fp, fn, tp = my_cm.ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

my_report = classification_report(y_true = y_test, y_pred = y_tahmin, labels=["no", "yes"])
print(my_report)