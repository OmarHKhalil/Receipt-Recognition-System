import joblib
import numpy as np

# تحميل الملف الموحد الذي يحتوي على (المودل، tfidf، و encoder)
model_data = joblib.load('model/MLP_Model.joblib')
mlp_model = model_data['model']
tfidf = model_data['tfidf']
encoder = model_data['encoder']

def get_model_prediction(text):
    # تحويل النص
    text_tfidf = tfidf.transform([text])
    # الحصول على الاحتمالات
    probs = mlp_model.predict_proba(text_tfidf)[0]
    best_idx = np.argmax(probs)
    confidence = probs[best_idx] * 100
    
    # الحصول على اسم الفئة
    try:
        category_name = encoder.classes_[best_idx]
    except AttributeError:
        category_name = encoder.categories_[0][best_idx]
        
    return category_name, confidence