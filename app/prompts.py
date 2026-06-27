ITEM_PROMPT = """
Return ONLY a JSON object. No intro, no outro, no markdown formatting.
Analyze this image as a product line item from an invoice and extract the data accurately.

Terms and Constraints:
1. Extraction: Extract the name (name), quantity (qty), unit price (unit_price), and total (total).
2. Classification (category_hint): Select the appropriate category exclusively from the following list:
['Beverages', 'Bakery & Breakfast', 'Snacks & Sweets', 'Home & Furniture', 'Fruits & Vegetables', 'Meat & Poultry', 'Household & Cleaning', 'Pantry & Cooking Ingredients', 'Dairy & Eggs', 'Personal Care', 'Baby Products', 'Office & Stationery', 'Electronics', 'Fast Food', 'Juices & Cocktails']
If no suitable category is found in the list, use the value "product".
3. Language (language):
If the name contains (Arabic only) or (both Arabic and English), write "Arabic".
If the name is (English only), write "English".
Required Output:
{
"type": "item",
"name": "string",
"qty": 0,
"unit_price": 0.0,
"total": 0.0,
"category_hint": "Selected Category",
"language": "Arabic/English"
}
"""

def get_meta_prompt(ctype):
    return f"""
Return ONLY a JSON object. No intro, no outro, no markdown formatting.
أنت خبير في معالجة الصور واستخراج النصوص (OCR) من الفواتير العربية.

المعطيات:
- نوع القصاصة الحالي من نظام YOLO هو: [{ctype}]

التعليمات الصارمة بناءً على التصنيف [{ctype}]:

1. إذا كان النوع (Merchant):
   -  ابحث عن أكبر نص في الصورة، أو أي نص داخل شعار (Logo). إذا كان الخط يدوياً أو مزخرفاً، خمن الاسم الأكثر منطقية.
   - صنف الـ "type" كـ "header".

2. إذا كان النوع (date) أو (number):
   - ركز على "date" وحوله لصيغة (YYYY-MM-DD).
   - ركز على "invoice_no" (رقم الفاتورة).
   - صنف الـ "type" كـ "header".

3. إذا كان النوع (total):
   - ركز فقط على القيمة العددية لـ "total". تجاهل العملة (ل.س، ريال، إلخ).
   - صنف الـ "type" كـ "footer".

قواعد عامة:
- لا تحاول اختراع بيانات غير موجودة في الصورة. إذا لم تجد الحقل المطلوب، ضعه null.
- التزم بصيغة JSON التالية حصراً:
{{
  "type": "header" | "footer",
  "shop_name": "string or null",
  "total": float or null,
  "date": "string or null",
  "invoice_no": "string or null"
}}
"""