# Audio Summarizer & Translator

## الوظيفة
مشروع ويب لتسجيل الصوت من المتصفح، ثم تلخيصه وترجمته تلقائياً.

## التشغيل المحلي
```bash
pip install -r requirements.txt
python app.py
```

## النشر على Railway
1. ارفع ملفات المشروع (app.py, requirements.txt, Procfile, templates/ ...إلخ) إلى GitHub أو Railway مباشرة.
2. Railway سيقرأ تلقائياً `requirements.txt` و `Procfile` ويشغل التطبيق.
3. إذا كان لديك متغيرات سرية (مثل مفاتيح API)، أضفها من إعدادات Railway > Variables.

## المتطلبات
- Python 3.9+
- اتصال إنترنت لتحميل النماذج

## الملفات الأساسية
- `app.py` : الكود الرئيسي
- `templates/index.html` : واجهة المستخدم
- `requirements.txt` : مكتبات بايثون المطلوبة
- `Procfile` : أمر التشغيل على Railway 