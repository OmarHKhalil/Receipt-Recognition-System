import cv2
from google.genai import types # استيراد الأنواع المطلوبة
from app.prompts import get_meta_prompt

def analyze_meta(crop, ctype, client):
    # 1. تحويل الصورة إلى بايتات
    _, buffer = cv2.imencode(".jpg", crop)
    image_bytes = buffer.tobytes()

    # 2. الحصول على البرومبت المناسب بناءً على النوع
    prompt = get_meta_prompt(ctype)

    # 3. بناء الطلب باستخدام التنسيق المتوافق مع المكتبة الجديدة
    # نستخدم gemini-2.0-flash لأنه المستقر حالياً ويدعم السرعة العالية
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                ]
            )
        ]
    )

    return response.text