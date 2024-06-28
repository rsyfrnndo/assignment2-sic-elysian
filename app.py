# Techinal Assignment 2
"""
Elysian Thinkers:   - Rasya Fernando
                    - Ahmad Fakhri A.
                    - Habibah Hisani N.
                    - Siti Lavifa N.
"""

#import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# membaca data
data = pd.read_csv("ai4i2020.csv")

# Ekplorasi data (menghapus dan mengklasifikasi biner)
for i in ["UDI", "Product ID", "Machine failure", "HDF", "PWF", "OSF", "RNF"]:
    data.drop(i, axis=1, inplace=True)

data = pd.concat([data.drop("Type", axis=1), pd.get_dummies(data['Type'], dtype='int')], axis=1)
X, y = data.drop("TWF", axis=1), data['TWF']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
train_data = X_train.join(y_train)

# visualisasi korelasi
plt.figure(figsize=(10,5))
sns.heatmap(train_data.corr(), annot=True, cmap="coolwarm")
plt.show()

"""

Ditemukan bahwa fitur yang memiliki korelasi tertinggi
dengan target TWF (Tool Wear Failure) adalah fitur
Penggunaan Tool wear (Tool Wear (min)) dengan korelasi
0.12

"""


# Menggunakan model Random Forest Regression
"""

Kita menggunakan model Random Forest karena 
kemampuannya dalam menangkap hubungan non-linear
dan dapan menangani fitur yang beragam.

"""

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R kuadrat: {r2}')

# Cek hasil prediksi
results = pd.DataFrame({'Data': y_test, 'Prediksi': y_pred})
print(results.head(100))