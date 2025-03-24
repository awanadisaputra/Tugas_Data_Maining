# Nama    : Awan Adi Saputra
# NIM     : F11.2023.00071

# Disini saya tidak pakai google colaps tapi menggunakan code editor
# Jika pakai code editor jangan lupa python harus sudah diinstall dilaptop
# Selain itu juga harus menginstall package yang dibutuhkan yaitu numpy, matplotlib, pandas, dan juga scikit-learn
# Cara menginstall packagenya menggunakan perintah pip3 install numpy pandas matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Mengimport dataset
file = "dataset.csv"
dataset = pd.read_csv(file)

# Menampilkan dataset asli
print("\n Dataset yang belum diapa-apakan : \n")
print(dataset)

# Disini karena dataset saya memiliki beberapa jenis inner data jadi menggunakan 3 jenis strategy
from sklearn.impute import SimpleImputer
imputerMean = SimpleImputer(strategy='mean')
imputerMedian = SimpleImputer(strategy='median')
imputerMostFrequent = SimpleImputer(strategy='most_frequent')

# Mengisi missing value
dataset[['Nilai', 'Jam_Tidur', 'Index_Performa']] = imputerMean.fit_transform(dataset[['Nilai', 'Jam_Tidur', 'Index_Performa']])
dataset[['Jam_Belajar', 'Jumlah_Latihan_Soal_Yang_Dikerjakan']] = imputerMedian.fit_transform(dataset[['Jam_Belajar', 'Jumlah_Latihan_Soal_Yang_Dikerjakan']])
dataset[['Ekstrakulikuler', 'Gender']] = imputerMostFrequent.fit_transform(dataset[['Ekstrakulikuler', 'Gender']])

# Menampilkan dataset setelah menghilangkan missing value
print("\n Dataset setelah menghilangkan missing valuec : \n")
print(dataset)

# Melakukan encoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
oEncoder = OneHotEncoder (drop='first', sparse_output=False)
gender = oEncoder.fit_transform(dataset[['Gender']])
dataset[['Gender']] = gender

print("\n Dataset setelah melakukan encoder pada coloum Gender : \n")
print(dataset)

encoder = LabelEncoder()
dataset['Ekstrakulikuler'] = encoder.fit_transform(dataset['Ekstrakulikuler'])

print("\n Dataset setelah melakukan encoder pada coloum Ekstrakulikuler : \n")
print(dataset)

# Membagi dataset menjadi training dan test
from sklearn.model_selection import train_test_split
x = dataset.drop(columns=['Index_Performa', 'Gender'])
y = dataset['Index_Performa']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\n Memunculkan data training dan test \n")
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# Melakukan feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train[['Jam_Belajar', 'Nilai', 'Jam_Tidur', 'Jumlah_Latihan_Soal_Yang_Dikerjakan']] = scaler.fit_transform(x_train[['Jam_Belajar', 'Nilai', 'Jam_Tidur', 'Jumlah_Latihan_Soal_Yang_Dikerjakan']])
x_test[['Jam_Belajar', 'Nilai', 'Jam_Tidur', 'Jumlah_Latihan_Soal_Yang_Dikerjakan']] = scaler.transform(x_test[['Jam_Belajar', 'Nilai', 'Jam_Tidur', 'Jumlah_Latihan_Soal_Yang_Dikerjakan']])

print("\n Memunculkan data yang telah difeature scaling \n")
print(x_train)
print(x_test)
