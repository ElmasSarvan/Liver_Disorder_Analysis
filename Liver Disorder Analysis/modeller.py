import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Gerekli Kütüphaneler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.utils import resample

# --- AYARLAR ---
file_path = "yeni_birlestirilmis_veri.csv"

print("--- KAPSAMLI VERİ ANALİZİ, GÖRSELLEŞTİRME VE MODELLEME ---\n")

# =============================================================================
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# =============================================================================
try:
    df = pd.read_csv(file_path)
    print(f"Veri Yüklendi. Boyut: {df.shape}")

    if 'Source' in df.columns:
        df = df.drop('Source', axis=1)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
except FileNotFoundError:
    print(f"HATA: Dosya bulunamadı: {file_path}")
    exit()

# Hedef Değişkeni Ayarlama
if 'Liver_Disease' in df.columns:
    df['Target'] = df['Liver_Disease'].apply(lambda x: 1 if x == 1 else 0)
    df = df.drop('Liver_Disease', axis=1)

# Target temizliği (Sadece 0 ve 1 kalsın)
if 'Target' in df.columns:
    if df['Target'].max() > 1:
        df['Target'] = df['Target'].map({1: 1, 2: 0})
    df = df.dropna(subset=['Target'])
    df['Target'] = df['Target'].astype(int)

# =============================================================================
# 2. ÖZELLİK MÜHENDİSLİĞİ (Feature Engineering)
# =============================================================================
sgot_col = 'Aspartate_Aminotransferase_Sgot' if 'Aspartate_Aminotransferase_Sgot' in df.columns else 'SGOT'
sgpt_col = 'Alamine_Aminotransferase_Sgpt' if 'Alamine_Aminotransferase_Sgpt' in df.columns else 'SGPT'

if sgot_col in df.columns and sgpt_col in df.columns:
    df['Enzim_Orani'] = df[sgot_col] / (df[sgpt_col] + 0.0001)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    print(">> Yeni Özellik: Enzim Oranı eklendi.")

if 'Albumin' in df.columns and 'Total_Protiens' in df.columns:
    df['Globulin'] = df['Total_Protiens'] - df['Albumin']
    df['A_G_Ratio_Calc'] = df['Albumin'] / (df['Globulin'] + 0.0001)
    print(">> Yeni Özellik: A/G Oranı (Hesaplanan) eklendi.")

# =============================================================================
# 3. VERİ GÖRSELLEŞTİRME (YENİ EKLENEN BÖLÜM)
# =============================================================================
print("\n--- Veri Görselleştirme İşlemleri Başlıyor ---")
custom_palette = {0: "green", 1: "red"}

# A. Korelasyon Matrisi
plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Özellikler Arası Korelasyon Matrisi')
plt.tight_layout()
plt.savefig('grafik_korelasyon_matrisi.png')
print(">> 'grafik_korelasyon_matrisi.png' kaydedildi.")

# B. Pairplot (Seçili Özellikler İçin)
# Çok fazla sütun olduğu için en önemlilerini seçiyoruz
pair_cols = ['Age', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Albumin', 'Enzim_Orani', 'Target']
plot_cols = [c for c in pair_cols if c in df.columns]

if len(plot_cols) > 1:
    sns.pairplot(df[plot_cols], hue='Target', palette=custom_palette, corner=True)
    plt.savefig('grafik_pairplot.png')
    print(">> 'grafik_pairplot.png' kaydedildi.")

# C. Dağılım Grafikleri (Distribution Plot)
dist_cols = ['Albumin', 'Alkaline_Phosphotase', 'Enzim_Orani']
valid_dist = [c for c in dist_cols if c in df.columns]
if valid_dist:
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(valid_dist):
        plt.subplot(1, 3, i + 1)
        sns.histplot(data=df, x=col, hue='Target', palette=custom_palette, kde=True, element="step")
        plt.title(f'{col} Dağılımı')
    plt.tight_layout()
    plt.savefig('grafik_yayilim_dagilimi.png')
    print(">> 'grafik_yayilim_dagilimi.png' kaydedildi.")

# D. Saçılım Grafiği (Scatter Plot)
if 'Age' in df.columns and 'Total_Bilirubin' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Age', y='Total_Bilirubin', hue='Target', palette=custom_palette, alpha=0.6)
    plt.title('Yaş vs Total Bilirubin')
    plt.savefig('grafik_scatter_yas_bilirubin.png')
    print(">> 'grafik_scatter_yas_bilirubin.png' kaydedildi.")

# =============================================================================
# 4. VERİ AYRIMI, DENGELEME VE ÖLÇEKLEME
# =============================================================================
X = df.drop('Target', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Eğitim verisini dengeleme (Up-sampling)
train_data = pd.concat([X_train, y_train], axis=1)
majority = train_data[train_data.Target == 0]
minority = train_data[train_data.Target == 1]

if len(minority) < len(majority):
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    train_balanced = pd.concat([majority, minority_upsampled])
else:
    majority_upsampled = resample(majority, replace=True, n_samples=len(minority), random_state=42)
    train_balanced = pd.concat([majority_upsampled, minority])

y_train_resampled = train_balanced.Target
X_train_resampled = train_balanced.drop('Target', axis=1)

# Ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 5. MODELLERİN EĞİTİMİ VE DETAYLI DEĞERLENDİRME
# =============================================================================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    "Support Vector Machine": SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
}

results = []
roc_data = {}

print("\n--- MODEL PERFORMANS RAPORU ---")

for name, model in models.items():
    print(f"\n>> {name} eğitiliyor...")
    model.fit(X_train_scaled, y_train_resampled)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    train_acc = accuracy_score(y_train_resampled, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_proba)

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_data[name] = (fpr, tpr, auc)

    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Recall:        {rec:.4f}")
    print(f"   ROC-AUC:       {auc:.4f}")

    results.append({
        'Model': name,
        'Train Acc': train_acc,
        'Test Acc': test_acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'AUC': auc
    })

    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sağlıklı', 'Hasta'], yticklabels=['Sağlıklı', 'Hasta'])
    plt.title(f'{name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
    plt.close()

# =============================================================================
# 6. ÖZELLİK ÖNEMİ GRAFİĞİ (FEATURE IMPORTANCE) - YENİ EKLENEN BÖLÜM
# =============================================================================
print("\n--- Özellik Önemi Analizi ---")
if "Random Forest" in models:
    rf_model = models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    # 'hue' uyarısını düzeltmek için hue parametresi eklendi ve legend kapatıldı
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, legend=False, palette='viridis')
    plt.title('Hastalık Teşhisinde En Etkili 10 Özellik (Random Forest)')
    plt.xlabel('Önem Derecesi')
    plt.tight_layout()
    plt.savefig('grafik_ozellik_onemi.png')
    print(">> 'grafik_ozellik_onemi.png' kaydedildi.")

# =============================================================================
# 7. SONUÇ GRAFİKLERİ (KARŞILAŞTIRMA)
# =============================================================================
results_df = pd.DataFrame(results)

# A. Overfitting Grafiği
plt.figure(figsize=(10, 6))
x = np.arange(len(results_df['Model']))
width = 0.35
plt.bar(x - width / 2, results_df['Train Acc'], width, label='Eğitim', color='skyblue')
plt.bar(x + width / 2, results_df['Test Acc'], width, label='Test', color='salmon')
plt.xticks(x, results_df['Model'])
plt.title('Model Başarı Karşılaştırması (Overfitting Kontrol)')
plt.legend()
plt.tight_layout()
plt.savefig('analiz_overfitting.png')

# B. Detaylı Metrikler
plt.figure(figsize=(12, 6))
metrics_to_plot = ['Test Acc', 'Precision', 'Recall', 'F1 Score', 'AUC']
results_df.set_index('Model')[metrics_to_plot].plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.title('Modellerin Detaylı Metrik Karşılaştırması')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('analiz_detayli_metrikler.png')

# C. ROC Eğrileri
plt.figure(figsize=(10, 8))
for name, (fpr, tpr, auc_score) in roc_data.items():
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi')
plt.legend(loc="lower right")
plt.savefig('analiz_roc_curve.png')

# =============================================================================
# 8. KAYIT
# =============================================================================
results_df['Score_Mean'] = (results_df['Test Acc'] + results_df['F1 Score'] + results_df['AUC']) / 3
best_model_row = results_df.loc[results_df['Score_Mean'].idxmax()]

print(f"\n========== SONUÇ ==========")
print(f"EN BAŞARILI MODEL: {best_model_row['Model']}")
print(f"Ortalama Skor: {best_model_row['Score_Mean']:.4f}")

joblib.dump(models[best_model_row['Model']], 'final_best_model.pkl')
joblib.dump(scaler, 'final_scaler.pkl')
print("\nİşlem Tamamlandı. Tüm grafikler kaydedildi.")