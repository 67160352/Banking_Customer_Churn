# 🏦 การทำนายลูกค้าที่เสี่ยงหนี

## 📌 โปรเจคนี้คืออะไร?

ใช้ **Machine Learning** ทำนายลูกค้าธนาคารที่จะปิดบัญชี (หนี) 
เพื่อให้ธนาคารสามารถ **ช่วยเหลือลูกค้าทันท่วงที**

---

## 🎯 ปัญหา

- 📊 ธนาคารมี 10,000 ลูกค้า
- 🚨 **20.4% (2,037 คน) ปิดบัญชีไปแล้ว**
- 💰 ต้นทุนหาลูกค้าใหม่ แพง 5-25 เท่า
- ❓ ไม่รู้ว่า**ใครจะหนี**ก่อนมันเกิด

## ✅ วิธีแก้

ทำนาย **ลูกค้าเสี่ยง** ให้ธนาคารติดต่อและเสนอ **ข้อเสนอพิเศษ** เพื่อรั้งไว้

---

## 📊 ข้อมูล

**Dataset:** ธนาคารยุโรป (10,000 ลูกค้า)

**ข้อมูลที่ใช้:**
- 👤 age (อายุ), gender (เพศ), geography (ประเทศที่อยู่)
- 💵 Estimated Salary (เงินเดือน), Balance (เงินในบัญชี), Credit Score (คะแนนเครดิตของลูกค้า)
- 🕐 Tenure(อยู่กับธนาคารกี่ปี)
- 📦 Products(ใช้บริการกี่ชนิด)
- ✅ บัญชีกำลังใช้งานไหม(IsActiveMember)
- 🎯 **Target:** ปิดบัญชีหรือไม่ (0/1) 1 = ปิดบัญชีไปแล้ว, 0 = เป็นลูกค้าอยู่

---
## ทำไม Machine Learning ถึงเหมาะกับปัญหานี้?

ปัจจัยที่ทำให้ลูกค้าปิดบัญชีมีความซับซ้อน เช่น "ลูกค้าอายุมาก + เงินในบัญชีสูง + ไม่ค่อยเคลื่อนไหวบัญชี" อาจจะมีโอกาสหนีสูง การใช้กฎ If-Else ธรรมดาไม่สามารถครอบคลุมทุกความน่าจะเป็นได้ การใช้ ML จะช่วยดึง Pattern ที่ซ่อนอยู่นี้ออกมาคำนวณเป็นเปอร์เซ็นต์ความเสี่ยง (Probability) ได้อย่างแม่นยำ
## 🔍 ผลการวิเคราะห์ข้อมูล

### ลูกค้าเสี่ยงหนีมีลักษณะนี้:
```
👴 อายุมาก        → 45 ปี (vs ที่อยู่ 37 ปี)
🆕 ลูกค้าใหม่      → 2 ปี (vs ที่อยู่ 3 ปี)
🇩🇪 จากเยอรมนี    → 32% หนี (vs ฝรั่งเศส 15%)
📦 ใช้บริการ 1 ชนิด → 27% หนี (vs 3 ชนิด 8%)
```

---

## 🤖 Model (โมเดล)

**Algorithm:** Gradient Boosting (ที่ดีที่สุด)

**ขั้นตอน:**
1. ✅ เตรียมข้อมูล (Preprocessing)
2. ✅ ฝึกสอนโมเดล (Training)
3. ✅ ปรับพารามิเตอร์ (Tuning)
4. ✅ ประเมินผล (Evaluation)

---

## 📈 ผลลัพธ์

| เมตริก | ค่า | ความหมาย |
|--------|-----|---------|
| **Accuracy** | 86.9% | ทั่วไป ถูก 87% |
| **Precision** | 81.4% | ถ้าบอกว่า "หนี" มี 81% ถูก |
| **Recall** | 46.2% | ⚠️ จับลูกค้าหนี 46% เท่านั้น |
| **ROC-AUC** | 0.87 | Discriminate ได้ดี |

### ✅ ดีตรงนี้:
- Accuracy 87% ✅
- Precision 81% ✅
- ROC-AUC 0.87 ✅

---

## 🚀 ใช้งาน

### วิธี 1: รันโปรแกรมทั้งหมด
```bash
python Banking_Customer_Churn.ipynb
```
ได้ผลลัพธ์ + กราฟ + บันทึกโมเดล

### วิธี 2: เปิด Web App
```bash
streamlit run churn_streamlit.py
```
เปิด http://localhost:8501
---
หรือกดเข้าลิ้งค์นี้ https://bankingcustomerchurn-vwzdcl7nvpemyypwqmg2hm.streamlit.app/#about-this-project
---
## 📱 Web App มี 4 Tab

### 🔮 **Tab 1: Predictor (ทำนาย)**
- ใส่ข้อมูลลูกค้า
- ได้ % โอกาสหนี
- ได้ Risk Level (Low/Medium/High)

### 📊 **Tab 2: Model Performance (ผลลัพธ์)**
- Metrics ทั้งหมด
- Confusion Matrix
- ROC Curve

### 📈 **Tab 3: Data Analysis (วิเคราะห์ข้อมูล)**
- Churn Distribution
- Age vs Churn
- Tenure vs Churn
- By Country

### ℹ️ **Tab 4: About (เกี่ยว)**
- อธิบายโปรเจค
- เทคโนโลยีที่ใช้

---

## 💡 ตัวอย่างการใช้

### ลูกค้า A: เสี่ยมสูง (80% หนี)
```
Age: 65 ปี (แก่)
Tenure: 1 ปี (ใหม่)
Country: Germany (เสี่ยง)
Products: 1 (ใช้แค่บัญชีเดียว)
IsActive: No (ไม่ใช้งาน)

→ ธนาคารต้องติดต่อ วันนี้!
→ เสนอสิ่งเสนอพิเศษ
```

### ลูกค้า B: เสี่ยมต่ำ (10% หนี)
```
Age: 35 ปี (หนุ่ม)
Tenure: 8 ปี (เก่า)
Country: France (ปลอดภัย)
Products: 3 (ใช้บริการหลาย)
IsActive: Yes (ใช้งาน)

→ ไม่ต้องห่วง ✅
→ ให้เสิร์วิสดี พอ
```

---

## 📁 ไฟล์ที่มี

```
Banking-Churn-Project/
├── churn_streamlit.py          ← Web app
├── Banking_Customer_Churn.ipynb      ← ML pipeline
├── Churn_Modelling.csv         ← ข้อมูล
├── requirements.txt      ← Library
└── README.md                   ← นี่เอง
```
---

## 🛠️ ติดตั้ง

```bash
# 1. ติดตั้ง Library
pip install -r requirements_churn.txt

# 2. รัน App
streamlit run churn_streamlit.py
```

---

## 🌐 Deploy บน Streamlit Cloud

```bash
# 1. Push ไป GitHub
git add .
git commit -m "Banking churn prediction"
git push origin main

# 2. ไปที่ streamlit.io/cloud
# 3. Create app → เลือก repo + churn_streamlit.py
# 4. Deploy!

→ ได้ URL: https://YOUR_USERNAME-banking-churn.streamlit.app
```

---

## 🎓 เรียนรู้อะไร

✅ Classification vs Regression (ต่างกันยังไง)
✅ Imbalanced Data (ข้อมูลไม่สมดุล)
✅ Feature Engineering (เลือกปัจจัย)
✅ Model Selection (เลือกโมเดล)
✅ Hyperparameter Tuning (ปรับพารามิเตอร์)
✅ Deployment (นำไปใช้จริง)
✅ Web Application (สร้าง App)

---

## ⚠️ ข้อควรจำ

### ❌ อันตราย
- **ไม่ใช่คำแนะนำทางการเงิน** เป็นแค่การศึกษา
- Model ใช้ข้อมูลเก่า อาจไม่ตรงกับอนาคต
- ต้องเรียนรู้ + อัปเดต อยู่เรื่อย

### ✅ ประโยชน์
- ช่วยธนาคาร **พบลูกค้าเสี่ยม** ก่อนหนี
- **ประหยัด** จากต้นทุนหาลูกค้าใหม่
- **พัฒนา** บริการให้ดีขึ้น

---

## 📌 สิ่งสำคัญ 3 ข้อ

1. **ลูกค้าที่เสี่ยง:** อายุแก่ + ใหม่ + อยู่เยอรมนี + บริการเดียว
2. **ทำนายได้:** 87% accuracy + 0.87 ROC-AUC
3. **นำไปใช้ได้:** Web app พร้อมใช้ + Deploy ไปที่ Streamlit Cloud
