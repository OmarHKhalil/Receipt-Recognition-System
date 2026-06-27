import cv2
from google.genai import types # مهم جداً للاستيراد
from app.prompts import ITEM_PROMPT

def analyze_item_generalized(crop, ctype, client):
    # 1. تحويل الصورة إلى bytes
    _, buffer = cv2.imencode(".jpg", crop)
    image_bytes = buffer.tobytes()

    # 2. بناء الطلب باستخدام التنسيق المتوافق مع Pydantic v2
    # يجب تغليف النص والصورة داخل كائنات types.Part
    response = client.models.generate_content(
        model="gemini-2.5-flash", # ملاحظة: تأكد إذا كان 2.0 أو 1.5 لأن 2.5 غير رسمي حالياً
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=ITEM_PROMPT),
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                ]
            )
        ]
    )
    
    return response.text