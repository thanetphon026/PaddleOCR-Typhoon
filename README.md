# 📦 Thai Parcel OCR System

ระบบสแกนและวิเคราะห์ข้อมูลพัสดุไทยอัตโนมัติ ด้วย PaddleOCR และ Typhoon API

## ✨ คุณสมบัติ

- 🔍 **OCR แม่นยำ**: ใช้ PaddleOCR สำหรับการอ่านข้อความไทย-อังกฤษ
- 🤖 **AI Extraction**: ใช้ Typhoon API วิเคราะห์และสกัดข้อมูลอัจฉริยะ
- ⚡ **รองรับ GPU/CPU**: ตรวจจับและใช้ GPU อัตโนมัติ (หากมี)
- 📊 **Performance Tracking**: แสดงเวลาประมวลผลแต่ละขั้นตอน
- 🎨 **Web Interface สวยงาม**: หน้าเว็บโมเดิร์นด้วย Dark Mode
- 📱 **Responsive Design**: ใช้งานได้ทั้ง Desktop และ Mobile

## 📋 ข้อมูลที่สกัดได้

1. **ชื่อผู้รับ** (Recipient Name)
2. **เลขห้อง** (Room Number)
3. **บริษัทขนส่ง** (Shipping Company)
4. **รหัสพัสดุ** (Tracking Number)

## 🔧 ความต้องการของระบบ

- **Python**: 3.9, 3.10 หรือ 3.11
- **RAM**: อย่างน้อย 4GB (แนะนำ 8GB)
- **GPU** (Optional): NVIDIA GPU with CUDA support

## 📥 การติดตั้ง

### 1. Clone โปรเจค

```bash
cd Desktop/paddleocr+typhhon
```

### 2. สร้าง Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. ติดตั้ง Dependencies

**สำหรับ CPU Only:**
```bash
pip install -r requirements.txt
```

**สำหรับ GPU (CUDA):**
```bash
# ถอนการติดตั้ง CPU version ก่อน
pip uninstall paddlepaddle -y

# ติดตั้ง GPU version (CUDA 11.8)
pip install paddlepaddle-gpu==2.6.0 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# หรือ CUDA 12.0
pip install paddlepaddle-gpu==2.6.0.post120 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

### 4. ตั้งค่า Environment Variables

สร้างไฟล์ `.env` จาก template:
```bash
copy .env.example .env
```

แก้ไขไฟล์ `.env` และใส่ Typhoon API Key:
```env
TYPHOON_API_KEY=your_actual_api_key_here
```

**วิธีการขอ Typhoon API Key:**
1. ไปที่: https://opentyphoon.ai/
2. สมัครสมาชิกและเข้าสู่ระบบ
3. ไปที่ API Keys section
4. สร้าง API key ใหม่
5. คัดลอกและใส่ในไฟล์ `.env`

## 🚀 การรันโปรแกรม

### เริ่มต้น Web Server

```bash
python app.py
```

### เข้าใช้งาน Web Interface

เปิดเบราว์เซอร์และไปที่:
```
http://localhost:5000
```

### ทดสอบระบบ

เช็คสถานะของระบบ:
```bash
curl http://localhost:5000/health
```

## 📖 วิธีใช้งาน

1. **เปิดเว็บเบราว์เซอร์** ไปที่ `http://localhost:5000`
2. **อัปโหลดรูปภาพ**: คลิกหรือลากรูพัสดุมาวาง
3. **คลิก "เริ่มสแกนพัสดุ"**: รอระบบประมวลผล
4. **ดูผลลัพธ์**: 
   - ชื่อผู้รับ
   - เลขห้อง
   - บริษัทขนส่ง
   - รหัสพัสดุ
   - เวลาที่ใช้แต่ละขั้นตอน

## ⏱️ Performance

### ค่าเฉลี่ยเวลาประมวลผล:

**CPU Mode:**
- PaddleOCR: 1.5 - 3.0 วินาที
- Typhoon API: 2.0 - 4.0 วินาที
- **รวม**: ~3.5 - 7.0 วินาที

**GPU Mode:**
- PaddleOCR: 0.3 - 0.8 วินาที
- Typhoon API: 2.0 - 4.0 วินาที
- **รวม**: ~2.3 - 4.8 วินาที

## 🏗️ โครงสร้างโปรเจค

```
paddleocr+typhhon/
│
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
│
├── modules/                   # Core modules
│   ├── __init__.py
│   ├── ocr_processor.py       # PaddleOCR integration
│   ├── typhoon_api.py         # Typhoon API client
│   └── utils.py               # Utility functions
│
├── static/                    # Static assets
│   ├── css/
│   │   └── style.css          # Styles
│   └── js/
│       └── main.js            # Frontend logic
│
├── templates/                 # HTML templates
│   └── index.html             # Main page
│
└── uploads/                   # Temporary uploads
```

## 🔍 การแก้ไขปัญหา

### ปัญหา: PaddleOCR ติดตั้งไม่สำเร็จ

**แก้ไข:**
```bash
pip install --upgrade pip setuptools wheel
pip install paddleocr==2.7.3 --no-cache-dir
```

### ปัญหา: OpenCV Error

**แก้ไข:**
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.9.0.80
```

### ปัญหา: Typhoon API Error 401

**แก้ไข:**
- ตรวจสอบว่าได้ใส่ API key ในไฟล์ `.env` ถูกต้อง
- ตรวจสอบว่า API key ยังใช้งานได้

### ปัญหา: GPU ไม่ทำงาน

**เช็ค CUDA:**
```python
import paddle
print(paddle.is_compiled_with_cuda())  # ควรได้ True
```

**ติดตั้ง CUDA Toolkit:**
- ดาวน์โหลดจาก: https://developer.nvidia.com/cuda-toolkit
- ติดตั้ง CUDA 11.8 หรือ 12.0

### ปัญหา: ไม่สามารถอ่านตัวอักษรไทยได้

**แก้ไข:**
- ตรวจสอบว่ารูปภาพชัดพอ (ขนาดอย่างน้อย 800x600 pixels)
- ลอง preprocessing รูปภาพ (contrast, brightness)
- ใช้ภาพที่มี resolution สูงกว่า

## 🎯 Optimization Tips

### 1. เพิ่มความเร็ว CPU
```python
# ใน ocr_processor.py
use_mp=True,              # เปิด multi-processing
enable_mkldnn=True,       # เปิด Intel MKL-DNN
total_process_num=4,      # ปรับตามจำนวน CPU cores
```

### 2. ลด Memory Usage
```python
# ใน ocr_processor.py
rec_batch_num=3,          # ลดจาก 6 เป็น 3
max_text_length=50,       # จำกัดความยาวข้อความ
```

### 3. เพิ่มความแม่นยำ
```python
# ใน ocr_processor.py
det_db_thresh=0.2,        # ลดเพื่อ detect ข้อความมากขึ้น
drop_score=0.2,           # ลดเพื่อยอมรับ confidence ต่ำลง
```

## 📝 License

MIT License - ใช้งานได้อย่างอิสระ

## 🙏 Credits

- **PaddleOCR**: https://github.com/PaddlePaddle/PaddleOCR
- **Typhoon API**: https://opentyphoon.ai/
- **Flask**: https://flask.palletsprojects.com/

## 📧 Support

หากมีปัญหาหรือข้อสงสัย:
1. ตรวจสอบ console log ใน terminal
2. ดู browser console (F12) สำหรับ frontend errors
3. เช็คไฟล์ `.env` ว่ามี API key ถูกต้อง

---

**Made with ❤️ for Thai Parcel Processing**
