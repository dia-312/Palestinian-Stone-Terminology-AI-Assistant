# 🪨 مساعد مصطلحات الحجر الفلسطيني

شات بوت ذكي متخصص في مصطلحات الحجر الفلسطيني المحلية، مبني على تقنية RAG (Retrieval-Augmented Generation).

## المميزات
- يجاوب على أسئلة مصطلحات الحجر المحلية الفلسطينية
- يستخدم قاعدة بيانات من المصطلحات المحلية
- يدعم المعرفة العامة بمجال الحجر والبناء
- واجهة شات سهلة الاستخدام

## المتطلبات
- Python 3.9+
- Gemini API Key من [Google AI Studio](https://aistudio.google.com)

## التثبيت

### 1. استنسخ المشروع
```bash
git clone  https://github.com/dia-312/Palestinian-Stone-Terminology-AI-Assistant.git
cd stone-rag
```

### 2. ثبّت المكتبات
```bash
pip install -r requirements.txt
```

### 3. أضف الـ API Key
انسخ الملف `.env.example` وسمّه `.env` ثم أضف الـ API key:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. أضف ملف المصطلحات
ضع ملف `terms.json` داخل مجلد `data/`

### 5. ابنِ قاعدة البيانات
```bash
python ingest.py
```

### 6. شغّل التطبيق
```bash
streamlit run app.py
```

## هيكل المشروع
```
stone_rag/
├── data/
│   └── terms.json        ← ملف المصطلحات
├── .env                  ← API Key (لا يُرفع على GitHub)
├── .env.example          ← مثال على الـ .env
├── .gitignore
├── requirements.txt
├── ingest.py             ← بناء قاعدة البيانات
├── app.py                ← واجهة Streamlit
└── README.md
```
