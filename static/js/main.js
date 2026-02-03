/**
 * Thai Parcel OCR System - Frontend JavaScript
 * Handles image upload, processing, and result display
 */

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const btnProcess = document.getElementById('btnProcess');
const loadingSection = document.getElementById('loading');
const resultsSection = document.getElementById('results');
const errorAlert = document.getElementById('errorAlert');
const errorMessage = document.getElementById('errorMessage');

// State
let selectedFile = null;

// ========== Event Listeners ==========

// Click to upload
uploadZone.addEventListener('click', () => {
    fileInput.click();
});

// File selection
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');

    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

// Process button
btnProcess.addEventListener('click', processImage);

// ========== Functions ==========

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    // Validate file
    if (!file) return;

    // Check file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
        showError('ประเภทไฟล์ไม่ถูกต้อง กรุณาเลือกไฟล์รูปภาพ (JPG, PNG, GIF, BMP, WebP)');
        return;
    }

    // Check file size (16MB max)
    const maxSize = 16 * 1024 * 1024; // 16MB
    if (file.size > maxSize) {
        showError('ไฟล์ใหญ่เกินไป (สูงสุด 16MB)');
        return;
    }

    // Store file
    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.classList.remove('hidden');
        hideError();
        hideResults();
    };
    reader.readAsDataURL(file);
}

/**
 * Process uploaded image
 */
async function processImage() {
    if (!selectedFile) {
        showError('กรุณาเลือกรูปภาพก่อน');
        return;
    }

    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);

    // Show loading
    showLoading();
    hideError();
    hideResults();
    btnProcess.disabled = true;

    try {
        // Send request
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Hide loading
        hideLoading();
        btnProcess.disabled = false;

        // Check response
        if (!response.ok || data.error) {
            showError(data.error || 'เกิดข้อผิดพลาดในการประมวลผล');
            return;
        }

        // Show results
        displayResults(data);

    } catch (error) {
        console.error('[ERROR] Fetch failed:', error);
        console.error('[ERROR] Details:', error.message);
        hideLoading();
        btnProcess.disabled = false;
        showError('เกิดข้อผิดพลาดในการเชื่อมต่อกับเซิร์ฟเวอร์: ' + error.message);
    }
}

/**
 * Display processing results
 */
function displayResults(data) {
    if (!data.success || !data.data) {
        showError('ไม่สามารถสกัดข้อมูลได้');
        return;
    }

    // Update result fields
    updateResultField('recipientName', data.data.recipient_name);
    updateResultField('roomNumber', data.data.room_number);
    updateResultField('shippingCompany', data.data.shipping_company);
    updateResultField('trackingNumber', data.data.tracking_number);

    // Update timing stats
    if (data.timings) {
        updateTiming('paddleTime', data.timings.paddle_ocr);
        updateTiming('typhoonTime', data.timings.typhoon_api);
        updateTiming('totalTime', data.timings.total);
    }

    // Show results section
    showResults();
}

/**
 * Update individual result field
 */
function updateResultField(elementId, value) {
    const element = document.getElementById(elementId);
    if (!element) return;

    if (!value || value === '' || value === 'ไม่พบข้อมูล' || value === 'ไม่สามารถสกัดข้อมูลได้') {
        element.textContent = 'ไม่พบข้อมูล';
        element.classList.add('empty');
    } else {
        element.textContent = value;
        element.classList.remove('empty');
    }
}

/**
 * Update timing display
 */
function updateTiming(elementId, seconds) {
    const element = document.getElementById(elementId);
    if (!element) return;

    if (typeof seconds === 'number') {
        element.textContent = `${seconds.toFixed(3)}s`;
    } else {
        element.textContent = '-';
    }
}

/**
 * Show loading animation
 */
function showLoading() {
    loadingSection.classList.add('active');
}

/**
 * Hide loading animation
 */
function hideLoading() {
    loadingSection.classList.remove('active');
}

/**
 * Show results section
 */
function showResults() {
    resultsSection.classList.add('active');
}

/**
 * Hide results section
 */
function hideResults() {
    resultsSection.classList.remove('active');
}

/**
 * Show error message
 */
function showError(message) {
    errorMessage.textContent = message;
    errorAlert.classList.add('active');
}

/**
 * Hide error message
 */
function hideError() {
    errorAlert.classList.remove('active');
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ========== Initialize ==========
console.log('🚀 Thai Parcel OCR System - Frontend Loaded');
