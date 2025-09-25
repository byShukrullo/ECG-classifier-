# Yurak kasalliklarini aniqlash uchun EKG (Elektrokardiogramma) rasmlarini klassifikatsiya qiluvchi AI modeli. @byshukrullo

## 📋 Umumiy Ma'lumot

Ushbu model EKG rasmlarini to'rt toifaga ajratuvchi TensorFlow/Keras asosidagi outputlarni taqdim etadi:

- **Normal** - Sog'lom yurak ritmi
- **mi (Miyokard infarkti)** - 
- **history_mi** - Oldingi yurak xuruji
- **abnormal** - Boshqa yurak anomaliyalari

## 🚀 Xususiyatlar

- ✅ Foydalanish oson Python interfeysi
- ✅ Barcha bashoratlar uchun ishonch darajasi (Accuracy)
- ✅ Batafsil ehtimollik taqsimoti (boshqa kategoriyalar o'rtasida)

## 📦 O'rnatish

### Oldindan Talab Qilinadigan Dasturlar

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pillow (PIL)

### O'rnatish

```bash
pip install tensorflow numpy pillow
```


## 🔧 Foydalanish

### Asosiy Foydalanish (Python Skript)

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('/Users/shukrullofoziljonov/Downloads/ecg_classifier_model.keras') #mendan olgan modelingizni yo'lini ko'rsating bu yerga
   
image_path = '/Users/shukrullofoziljonov/Downloads/ECG_DATA/train/mi/MI(1) - Copy - Copy.jpg'   #tahlil qilmoqchi bo'lgan rasmingizni yo'li

img = image.load_img(image_path, target_size=(128, 128)) 
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) 
img_array /= 255.0  
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions, axis=1)[0]
confidence = np.max(predictions, axis=1)[0] * 100
class_names = ["abnormal", "history_mi", "mi", "normal"]
predicted_class = class_names[predicted_class_index]

print("\n NATIJA ")
print(f"Bashorat: {predicted_class}")
print(f"Aniqlilik: {confidence:.2f}%")
print("\nBarcha sinflardagi ehtimolligi:")
for i, prob in enumerate(predictions[0]):
    print(f"- {class_names[i]}: {prob * 100:.2f}%") 
```

## 🧠 Model Arxitekturasi

- **Input**: 128x128 RGB rasmlar (Agar rasmingiz bu formatda bo'lmasa ham model avtomatik to'g'irlab oladi)
- **Sinflar**: 4 ta kategoriya (normal, mi, history_mi, abnormal)
- **Framework**: TensorFlow/Keras
- **Output**: Barcha sinflar bo'yicha ehtimollik taqsimoti

## 📸 Natija Namunasi

```
NATIJA 
Bashorat: mi
Aniqlilik: 100.00%

Barcha sinflardagi ehtimolligi:
- abnormal: 0.00%
- history_mi: 0.00%
- mi: 100.00%
- normal: 0.00%
```

## 🔬 Ma'lumotlar Talablari

### Rasm Formati
- **Qo'llab-quvvatlanadigan formatlar**: JPG, JPEG, PNG, BMP
- **Rang**: RGB (3 kanal)

### Ma'lumotlarni Qayta Ishlashdagi fzalliklari
- Rasmlar avtomatik ravishda [0,1] oralig'iga normallanadi
- Model kirish o'lchamlariga qayta o'lchamlanadi
- Qo'shimcha qayta ishlash talab qilinmaydi


## 🚨 Muhim Eslatmalar

### Tibbiy Ogohlantirish
⚠️ **Ushbu model faqat tadqiqot va ta'lim maqsadlari uchun mo'ljallangan. Haqiqiy tibbiy tashxis qo'yish uchun tibbiyot mutaxassislarining tasdiqi va tekshiruvidan o'tmasdan foydalanilmasligi kerak.**

### Sog'liqni saqlash vazirligi yoki Xaqlaro tibbiy tashkilotlar tomonidan tasdiqlanmagan yoki klinik jihatdan tekshirilmagan!

