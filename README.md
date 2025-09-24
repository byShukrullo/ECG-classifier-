# EKG Tasvirlarini Tasniflash Modeli

Bu loyiha EKG (elektrokardiogramma) tasvirlarini tahlil qilish va ularni toʻrtta asosiy sinfga (normal-nomal, mi-Miokard infarkti, abnormal-abnormal yurak urushi, history_mi-Miokard infarkti tarixiga ega) tasniflash uchun mo'ljallangan konvolyutsion neyron tarmog'i (CNN) modelini o'z ichiga oladi.

## Loyiha Maqsadi

Ushbu model EKG rasmlariga asoslangan holda avtomatik ravishda yurak holatini (masalan, yurak xuruji yoki boshqa anomaliyalar) aniqlash uchun yaratilgan. Model TensorFlow va Keras yordamida o'rgatilgan.

## Model Arxitekturasi

Model uchta Konvolyutsion (Conv2D) qatlamdan iborat bo'lib, ular har bir bosqichda EKG tasvirlaridan tobora murakkab xususiyatlarni ajratib oladi.

| Qatlam Turi | Filtrlar Soni | Chiqish O'lchami | Vazn (Parametrlar) Soni |
| :--- | :--- | :--- | :--- |
| **Conv2D** | 32 | (128, 128, 32) | 896 |
| **MaxPooling2D** | - | (64, 64, 32) | 0 |
| **Conv2D** | 64 | (64, 64, 64) | 18496 |
| **MaxPooling2D** | - | (32, 32, 64) | 0 |
| **Conv2D** | 128 | (32, 32, 128) | 73856 |
| **MaxPooling2D** | - | (16, 16, 128) | 0 |
| **Flatten** | - | (32768) | 0 |
| **Dense** | 512 | (512) | 16777728 |
| **Dense** | 4 | (4) | 2052 |

- **Jami Parametrlar**: ~16.9 million
- **O'rganiladigan Parametrlar**: ~16.9 million
- **O'rganilmaydigan Parametrlar**: 0

Model **Adam** optimizatori va **`categorical_crossentropy`** yo'qotish funksiyasi yordamida 10 epoch davomida o'qitilgan.

## Loyihani Ishga Tushirish

### Talablar

Ushbu loyihani ishga tushirish uchun quyidagi kutubxonalar o'rnatilgan bo'lishi kerak:
- `tensorflow`
- `keras`
- `numpy`
- `Pillow` (yoki `PIL`)

Bularni o'rnatish uchun quyidagi buyruqni ishga tushiring:
```bash
pip install tensorflow keras numpy pillow

