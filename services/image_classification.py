import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

device = torch.device("cpu")

def load_resnet_model(model_path):
    # نستخدم ResNet50
    model = models.resnet50(weights=None) 
    num_ftrs = model.fc.in_features
    # الموديل مدرب على 3 كلاسات (invoice, menu, others)
    model.fc = nn.Linear(num_ftrs, 3) 
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

resnet_model = load_resnet_model("model/invoice_classification.pth")

# الإعدادات التي استخدمتها أنت في التدريب (يجب أن تكون متطابقة 100%)
test_transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    # تحويل لرمادي مع الحفاظ على 3 قنوات ليقبلها ResNet
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def is_invoice(cv2_img):
    # تحويل من BGR (OpenCV) إلى RGB (PIL)
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # تطبيق التحويلات (Resize 448 + Grayscale)
    input_tensor = test_transforms(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = resnet_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    predicted_idx = preds.item()
    confidence = probabilities[0][predicted_idx].item() * 100

    # طباعة التفاصيل للتأكد من سير العمل
    print(f"--- [ResNet Filter] ---")
    print(f"Detected Category: {predicted_idx} (0=Invoice, 1=Menu, 2=Other)")
    print(f"Confidence: {confidence:.2f}%")
    
    # إرجاع True فقط إذا كان التصنيف هو 0 (Invoice)
    return predicted_idx == 0