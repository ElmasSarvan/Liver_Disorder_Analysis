import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. VERİ YÜKLEME
# ==========================================
print("Dosyalar yükleniyor...")

# ILPD.csv (Header yok, bu yüzden header=None diyoruz ve isimleri elle veriyoruz)
try:
    ilpd = pd.read_csv('ILPD.csv', header=None)
    ilpd.columns = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A_G_Ratio', 'Target']
    # Target düzeltme (2 -> 0 yapalım, 1 -> 1 kalsın)
    ilpd['Target'] = ilpd['Target'].map({1: 1, 2: 0})
    print(f"ILPD yüklendi: {ilpd.shape}")
except FileNotFoundError:
    print("UYARI: ILPD.csv bulunamadı, bu adım atlanıyor.")
    ilpd = pd.DataFrame()

# Kaggle Verisi
try:
    kaggle = pd.read_csv('Liver_disease_data.csv')
    # Sütun ismi uyumluluğu için yeniden adlandırma (varsa)
    kaggle.rename(columns={'Diagnosis': 'Target', 'Dataset': 'Target'}, inplace=True)
    print(f"Kaggle verisi yüklendi: {kaggle.shape}")
except FileNotFoundError:
    print("UYARI: Liver_disease_data.csv bulunamadı.")
    kaggle = pd.DataFrame()

if ilpd.empty and kaggle.empty:
    print("HATA: Hiçbir veri dosyası bulunamadı!")
    exit()

# ==========================================
# 2. BİRLEŞTİRME (Concatenation)
# ==========================================
# İki veri setini alt alta ekliyoruz. Ortak olmayan sütunlar NaN (boş) olacaktır.
df = pd.concat([ilpd, kaggle], axis=0, ignore_index=True)
print(f"Birleştirme tamamlandı. Yeni boyut: {df.shape}")

# ==========================================
# 3. VERİ TEMİZLİĞİ VE DÜZENLEME
# ==========================================

# Cinsiyet (Gender) Düzenleme: Male/Female -> 1/0
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].astype(str).str.strip() # Boşlukları temizle
    gender_map = {'Male': 1, 'Female': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0, 'nan': 0}
    df['Gender'] = df['Gender'].map(gender_map).fillna(0).astype(int)

# Eksik Verileri Doldurma (Sadece sayısal sütunlar)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'Target': # Hedef sütunu doldurma
        df[col] = df[col].fillna(df[col].median())

# Target sütunundaki boşlukları (NaN) temizle
if 'Target' in df.columns:
    df = df.dropna(subset=['Target'])
    df['Target'] = df['Target'].astype(int)

# ==========================================
# 4. DOSYAYI KAYDETME (İŞTE EKSİK OLAN KISIM)
# ==========================================
output_filename = 'yeni_birlestirilmis_veri.csv'
df.to_csv(output_filename, index=False)
print(f"\n>>> BAŞARILI! Dosya '{output_filename}' adıyla kaydedildi.")

# ==========================================
# 5. MODEL HAZIRLIĞI (Opsiyonel Kontrol)
# ==========================================
X = df.drop('Target', axis=1)
y = df['Target']

# Hata kontrolü için basit bir standartlaştırma denemesi
try:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Veri standartlaştırma testi başarılı (Model için hazır).")
except Exception as e:
    print(f"Standartlaştırma uyarısı: {e}")
    print("Not: Metin içeren sütunlar (Object) hala var olabilir, kontrol edin.")