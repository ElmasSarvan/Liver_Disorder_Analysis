import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
import os
import numpy as np

# --- AYARLAR ---
MODEL_PATH = 'final_best_model.pkl'
SCALER_PATH = 'final_scaler.pkl'


def load_assets():
    """Model ve ölçekleyiciyi yükler."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        messagebox.showerror("Kritik Hata",
                             f"Model dosyaları bulunamadı:\n{MODEL_PATH}\n{SCALER_PATH}\n\nLütfen önce 'madencilik.py' analiz kodunu çalıştırın.")
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        messagebox.showerror("Hata", f"Model yüklenirken hata oluştu:\n{e}")
        return None, None


# Global Model Yüklemesi
MODEL, SCALER = load_assets()


class PredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("Karaciğer Hastalığı Teşhis Sistemi (Güncel Veri Seti)")
        master.geometry("600x850")

        if MODEL is None:
            master.destroy()
            return

        # Başlık
        title_label = tk.Label(master, text="Hasta Veri Girişi", font=("Arial", 16, "bold"), fg="#2c3e50")
        title_label.pack(pady=15)

        # Giriş Alanları Çerçevesi
        main_frame = tk.Frame(master)
        main_frame.pack(pady=5, padx=20, fill="both", expand=False)

        # -- GİRİŞ ALANLARI (Yeni Veri Setine Göre Düzenlendi) --
        # Sütun Eşleştirmeleri: 'Etiket': 'CSV_Sütun_Adı'
        self.input_fields = {
            "Yaş": "Age",
            "Cinsiyet (1: Erkek, 0: Kadın)": "Gender",
            "Vücut Kitle İndeksi (BMI)": "BMI",
            "Alkol Tüketimi (Ünite/Hafta)": "AlcoholConsumption",
            "Sigara (1: Evet, 0: Hayır)": "Smoking",
            "Genetik Risk (0: Düşük, 1: Orta, 2: Yüksek)": "GeneticRisk",
            "Fiziksel Aktivite (Saat/Hafta)": "PhysicalActivity",
            "Diyabet (1: Var, 0: Yok)": "Diabetes",
            "Hipertansiyon (1: Var, 0: Yok)": "Hypertension",
            "Karaciğer Fonksiyon Testi (Skor)": "LiverFunctionTest",

            # Biyokimya Değerleri (Kısa Kodlar)
            "SGPT (ALT)": "Sgpt",
            "SGOT (AST)": "Sgot",
            "Total Bilirubin (TB)": "TB",
            "Direct Bilirubin (DB)": "DB",
            "Alkalin Fosfataz (Alkphos)": "Alkphos",
            "Total Protein (TP)": "TP",
            "Albumin (ALB)": "ALB",
            "A/G Oranı": "A_G_Ratio"
        }

        self.entries = {}
        row = 0
        for label_text, feature_key in self.input_fields.items():
            lbl = tk.Label(main_frame, text=f"{label_text}:", font=("Arial", 10, "bold"), fg="#34495e")
            lbl.grid(row=row, column=0, sticky="e", padx=10, pady=3)

            entry = tk.Entry(main_frame, width=25, font=("Arial", 10))
            entry.grid(row=row, column=1, padx=10, pady=3)
            entry.insert(0, "0")  # Varsayılan 0
            self.entries[feature_key] = entry
            row += 1

        # Buton Alanı
        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=20)

        self.predict_button = tk.Button(btn_frame, text="RİSKİ HESAPLA", command=self.predict,
                                        bg="#2980b9", fg="white", font=("Arial", 12, "bold"),
                                        height=2, width=30, relief="raised", cursor="hand2")
        self.predict_button.pack()

        # Sonuç Alanı
        self.result_label = tk.Label(master, text="Verileri girip butona basınız.", font=("Arial", 12))
        self.result_label.pack(pady=10)

    def predict(self):
        input_data = {}

        # 1. VERİ TOPLAMA
        for label, key in self.input_fields.items():
            raw_val = self.entries[key].get().strip().replace(',', '.')  # Virgül düzeltme
            if not raw_val:
                messagebox.showwarning("Eksik Veri", f"Lütfen '{label}' alanını doldurunuz.")
                return
            try:
                input_data[key] = float(raw_val)
            except ValueError:
                messagebox.showerror("Hata", f"'{label}' için geçerli bir sayı giriniz.")
                return

        try:
            # 2. DataFrame Oluştur
            df = pd.DataFrame([input_data])

            # 3. FEATURE ENGINEERING (Özellik Mühendisliği)
            # Yeni veri setindeki sütun isimlerine (Sgot, TP, ALB vb.) göre hesaplama

            # A. Enzim Oranı (SGOT / SGPT)
            sgot = df.get('Sgot', 0)
            sgpt = df.get('Sgpt', 0)
            if isinstance(sgot, pd.Series): sgot = sgot.iloc[0]
            if isinstance(sgpt, pd.Series): sgpt = sgpt.iloc[0]

            df['Enzim_Orani'] = sgot / (sgpt + 0.0001)

            # B. Globulin ve A/G Ratio Calc
            # Globulin = Total Protein (TP) - Albumin (ALB)
            tp = df.get('TP', 0)
            alb = df.get('ALB', 0)
            if isinstance(tp, pd.Series): tp = tp.iloc[0]
            if isinstance(alb, pd.Series): alb = alb.iloc[0]

            df['Globulin'] = tp - alb
            df['A_G_Ratio_Calc'] = alb / (df['Globulin'] + 0.0001)

            # Sonsuz değer temizliği
            df.replace([np.inf, -np.inf], 0, inplace=True)

            # 4. SÜTUN HİZALAMA (Otomatik Eşleştirme)
            # Modelin eğitildiği sütunları alıyoruz.
            # Eğer modelinizde 'Drinks', 'MCV' gibi eski sütunlar kaldıysa,
            # aşağıdaki kod onları otomatik olarak 0 ile doldurur.

            if hasattr(SCALER, 'feature_names_in_'):
                model_cols = SCALER.feature_names_in_
            else:
                # Yedek liste (Model dosyasından okuyamazsa)
                model_cols = [
                    'Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB',
                    'A_G_Ratio', 'BMI', 'AlcoholConsumption', 'Smoking', 'GeneticRisk',
                    'PhysicalActivity', 'Diabetes', 'Hypertension', 'LiverFunctionTest',
                    'Enzim_Orani', 'Globulin', 'A_G_Ratio_Calc'
                ]

            # Eksik sütunları 0 ile doldur
            for col in model_cols:
                if col not in df.columns:
                    df[col] = 0

            # DataFrame'i modelin istediği sıraya sok
            df_final = df[model_cols]

            # 5. TAHMİN
            scaled_data = SCALER.transform(df_final)
            prediction = MODEL.predict(scaled_data)[0]

            if hasattr(MODEL, "predict_proba"):
                prob = MODEL.predict_proba(scaled_data)[0][1] * 100
            else:
                prob = 0

            # 6. SONUÇ
            if prediction == 1:
                res_text = f"🔴 SONUÇ: HASTA (Risk: %{prob:.1f})"
                color = "#c0392b"
            else:
                res_text = f"🟢 SONUÇ: SAĞLIKLI (Risk: %{prob:.1f})"
                color = "#27ae60"

            self.result_label.config(text=res_text, fg=color, font=("Arial", 16, "bold"))

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            messagebox.showerror("Analiz Hatası", f"Beklenmeyen bir hata oluştu:\n{e}\n\nDetay:\n{err_msg}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()