"""
Thai Parcel OCR System - Main Flask Application
Uses PaddleOCR for text extraction and Typhoon API for intelligent data parsing
"""

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import time
from dotenv import load_dotenv
from modules.ocr_processor import OCRProcessor
from modules.typhoon_api import TyphoonAPI
from modules.utils import allowed_file, cleanup_old_files

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize processors
# force_cpu=False = Auto-detect GPU/CPU (will use GPU if available and CUDA is properly installed)
# force_cpu=True = Force CPU mode only
ocr_processor = OCRProcessor(force_cpu=False)  # Auto-detect mode
typhoon_api = TyphoonAPI()

@app.route('/')
def index():
    """Render the main web interface"""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_parcel():
    """
    Process uploaded parcel image
    Returns: JSON with extracted data and timing metrics
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'ไม่มีไฟล์รูปภาพ'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'ประเภทไฟล์ไม่ถูกต้อง (รองรับ: jpg, jpeg, png)'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize timing
        total_start = time.time()
        timings = {}
        
        # Step 1: PaddleOCR Processing
        print(f"[1/2] Processing OCR for: {filename}")
        ocr_start = time.time()
        ocr_text = ocr_processor.extract_text(filepath)
        ocr_time = time.time() - ocr_start
        timings['paddle_ocr'] = round(ocr_time, 3)
        
        if not ocr_text or len(ocr_text.strip()) < 5:
            cleanup_old_files(app.config['UPLOAD_FOLDER'])
            return jsonify({
                'error': 'ไม่สามารถอ่านข้อความจากภาพได้',
                'timings': timings
            }), 400
        
        print(f"✓ OCR completed in {ocr_time:.3f}s")
        print(f"Extracted text preview: {ocr_text[:100]}...")
        
        # Step 2: Typhoon API Processing
        print(f"[2/2] Analyzing with Typhoon API...")
        typhoon_start = time.time()
        extracted_data = typhoon_api.extract_parcel_data(ocr_text)
        typhoon_time = time.time() - typhoon_start
        timings['typhoon_api'] = round(typhoon_time, 3)
        
        print(f"✓ Typhoon API completed in {typhoon_time:.3f}s")
        
        # Calculate total time
        total_time = time.time() - total_start
        timings['total'] = round(total_time, 3)
        
        # Cleanup old files
        cleanup_old_files(app.config['UPLOAD_FOLDER'], max_age_minutes=30)
        
        # Prepare response
        response = {
            'success': True,
            'data': extracted_data,
            'timings': timings,
            'raw_text_preview': ocr_text[:200] if len(ocr_text) > 200 else ocr_text
        }
        
        print(f"✓ Total processing time: {total_time:.3f}s")
        print(f"Results: {extracted_data}")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"✗ Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': f'เกิดข้อผิดพลาด: {str(e)}',
            'timings': timings if 'timings' in locals() else {}
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ocr_ready': ocr_processor.is_ready(),
        'typhoon_api_configured': typhoon_api.is_configured()
    }), 200

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Thai Parcel OCR System Starting...")
    print("=" * 60)
    
    # Show device info
    device_info = ocr_processor.get_device_info()
    device_icon = "🎮" if device_info['device'] == 'GPU' else "💻"
    print(f"OCR Engine: {'Ready ✓' if ocr_processor.is_ready() else 'Not Ready ✗'}")
    print(f"OCR Device: {device_icon} {device_info['device']} Mode")
    print(f"Typhoon API: {'Configured ✓' if typhoon_api.is_configured() else 'Not Configured ✗'}")
    print(f"Upload Folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print("=" * 60)
    print("📱 Access the web interface at: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
