import asyncio
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import numpy as np
import cv2
from google import genai 

# استيراد الخدمات الخاصة بك
from services.yolo_service import detect_and_crop
from services.item_api import analyze_item_generalized
from services.meta_api import analyze_meta
from utils.json_utils import safe_json
from services.Information_Extraction import get_model_prediction
from services.image_classification import is_invoice
from app.config import GEMINI_API_KEY_META1, GEMINI_API_KEY_META2, GEMINI_API_KEY_ITEMS1, GEMINI_API_KEY_ITEMS2
from app.config import GEMINI_API_KEY_ITEMS3, GEMINI_API_KEY_ITEMS4, GEMINI_API_KEY_ITEMS5
# 1. تعريف طوابير المهام
item_queue = asyncio.Queue()
meta_queue = asyncio.Queue()

# 2. دالة العامل المحسنة برمجياً (تمنع تجميد الـ Threads أثناء النوم)
async def api_worker(worker_id, api_func, queue, api_key, sleep_time):
    print(f"✅ {worker_id} online and ready.", flush=True)
    
    try:
        # إنشاء العميل مرة واحدة لكل عامل خارج الـ Loop لتوفر الاستهلاك
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"❌ {worker_id} failed to initialize client: {e}", flush=True)
        return

    while True:
        # سحب المهمة من الطابور
        task_data, future = await queue.get()
        ctype = task_data["type"]
        crop = task_data["image"]
        
        print(f"🚀 {worker_id} | Working on [{ctype}]...", flush=True)
        start_process = time.time()

        try:
            # تشغيل الدالة والانتظار بطريقة غير معطلة (Non-blocking)
            result = await asyncio.to_thread(api_func, crop, ctype, client)

            if not future.done():
                future.set_result((ctype, result))
            
            elapsed = time.time() - start_process
            print(f"✨ {worker_id} | Finished [{ctype}] in {elapsed:.2f}s.", flush=True)

        except Exception as e:
            print(f"❌ {worker_id} | Error: {e}", flush=True)
            if not future.done():
                future.set_exception(e)
        finally:
            queue.task_done()
            
            # تنفيذ النوم الإجباري بطريقة آمنة (ياحبذا لو كان النوم خارج الـ Thread)
            if sleep_time > 0:
                print(f"💤 {worker_id} | Sleeping for {sleep_time}s to preserve quota...", flush=True)
                # استخدام asyncio.sleep يضمن تحرير الـ Event Loop لكي يستجيب السيرفر للطلبات الأخرى
                await asyncio.sleep(sleep_time)

# 3. إدارة دورة حياة التطبيق (Lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    all_workers = []
    
    print("🛠️ Starting Workers...", flush=True)
    
    # توزيع عمال العناصر (Items) - 8 عمال
    for i in range(1, 3):
        all_workers.append(asyncio.create_task(
            api_worker(f"W_Item_K1_{i}", analyze_item_generalized, item_queue, GEMINI_API_KEY_ITEMS1, 60)))
    for i in range(1, 3):
        all_workers.append(asyncio.create_task(
            api_worker(f"W_Item_K2_{i}", analyze_item_generalized, item_queue, GEMINI_API_KEY_ITEMS2, 60)))
    for i in range(1, 3):
        all_workers.append(asyncio.create_task(
            api_worker(f"W_Item_K3_{i}", analyze_item_generalized, item_queue, GEMINI_API_KEY_ITEMS3, 60)))
    for i in range(1, 3):
        all_workers.append(asyncio.create_task(
            api_worker(f"W_Item_K4_{i}", analyze_item_generalized, item_queue, GEMINI_API_KEY_ITEMS4, 60)))
    # توزيع عمال الميتا (Meta) - 4 عمال
    for i in range(1, 3):
        all_workers.append(asyncio.create_task(
            api_worker(f"W_Meta_K1_{i}", analyze_meta, meta_queue, GEMINI_API_KEY_META1, 60)))
    for i in range(1, 3):
        all_workers.append(asyncio.create_task(
            api_worker(f"W_Meta_K2_{i}", analyze_meta, meta_queue, GEMINI_API_KEY_META2, 60)))
    # إعطاء فرصة بسيطة للعمال للبدء
    # إعطاء فرصة بسيطة للعمال للبدء
    await asyncio.sleep(0.5)
    print("🚀 All workers are active.")

    yield
    # إغلاق العمال عند إيقاف السيرفر
    for w in all_workers:
        w.cancel()

app = FastAPI(lifespan=lifespan)

# 4. نقطة النهاية لمعالجة الفاتورة
@app.post("/process")
async def process(file: UploadFile = File(...)):
    start_time = time.time()
    
    # قراءة الصورة
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    # التحقق من نوع الفاتورة
    if not await asyncio.to_thread(is_invoice, image):
        raise HTTPException(status_code=400, detail="Invalid Invoice")
    
    # YOLO: الكشف والقص
    crops = detect_and_crop(image)
    if not crops:
        return {"error": "No items detected"}

    futures = []
    
    # توزيع القصاصات على الطوابير المناسبة
    for c in crops:
        fut = asyncio.get_running_loop().create_future()
        target_queue = item_queue if c["type"] == "item" else meta_queue
        await target_queue.put(({"image": c["image"], "type": c["type"]}, fut))
        futures.append(fut)

    print(f"📥 Queued {len(futures)} tasks. Waiting for workers...", flush=True)

    # الانتظار حتى تنتهي جميع المهام
    try:
        # إضافة Timeout إجمالي (30 ثانية) لضمان عدم التعليق للأبد
        results = await asyncio.wait_for(asyncio.gather(*futures), timeout=660)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing Timeout")
    
    # تجميع النتائج النهائية
    final_response = {"header": {}, "items": [], "footer": {}}
    for ctype, result in results:
        data = safe_json(result)
        if not data: 
            continue
            
        if ctype == "item":
            name = data.get("name", "")
            # تصنيف المنتج
            cat, conf = get_model_prediction(name) if name else (None, 0)
            if conf > 87: 
                data["category"] = cat
            final_response["items"].append(data)
        else:
            # دمج بيانات الهيدر والفوتر
            for k in ["shop_name", "date", "invoice_no", "address"]:
                if k in data and data[k]:
                    final_response["header"][k] = data[k]
            for k in ["total", "tax", "payment_method"]:
                if k in data and data[k]:
                    final_response["footer"][k] = data[k]

    # تصحيح المجموع تلقائياً إذا فُقد
    if not final_response["footer"].get("total"):
        try:
            final_response["footer"]["total"] = sum(float(str(i.get("total", 0)).replace(',', '')) for i in final_response["items"])
        except:
            final_response["footer"]["total"] = 0

    print(f"🏁 Done in: {time.time() - start_time:.2f}s", flush=True)
    return final_response