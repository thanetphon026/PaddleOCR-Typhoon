"""
Utility Functions - Helper tools for the Thai Parcel OCR System
"""

import os
import time
from datetime import datetime, timedelta
from typing import Set

# Allowed image extensions
ALLOWED_EXTENSIONS: Set[str] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed
    
    Args:
        filename: Name of file to check
        
    Returns:
        bool: True if extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files(directory: str, max_age_minutes: int = 30) -> int:
    """
    Clean up old uploaded files from directory
    
    Args:
        directory: Directory to clean
        max_age_minutes: Maximum age of files to keep (default: 30 minutes)
        
    Returns:
        int: Number of files deleted
    """
    try:
        if not os.path.exists(directory):
            return 0
        
        now = time.time()
        max_age_seconds = max_age_minutes * 60
        deleted_count = 0
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            # Skip if not a file
            if not os.path.isfile(filepath):
                continue
            
            # Check file age
            file_age = now - os.path.getmtime(filepath)
            
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                    print(f"Deleted old file: {filename}")
                except Exception as e:
                    print(f"Failed to delete {filename}: {str(e)}")
        
        return deleted_count
        
    except Exception as e:
        print(f"Cleanup error: {str(e)}")
        return 0

def format_timing(seconds: float) -> str:
    """
    Format timing in human-readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"

def timing_decorator(func):
    """
    Decorator to measure function execution time
    
    Usage:
        @timing_decorator
        def my_function():
            pass
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"⏱ {func.__name__} took {format_timing(elapsed)}")
        
        return result
    
    return wrapper

def ensure_dir_exists(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Directory path to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def get_file_size_mb(filepath: str) -> float:
    """
    Get file size in megabytes
    
    Args:
        filepath: Path to file
        
    Returns:
        float: File size in MB
    """
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    return 0.0

def validate_image_file(filepath: str) -> tuple[bool, str]:
    """
    Validate image file
    
    Args:
        filepath: Path to image file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check if file exists
    if not os.path.exists(filepath):
        return False, "ไฟล์ไม่พบ"
    
    # Check file size (max 16MB)
    size_mb = get_file_size_mb(filepath)
    if size_mb > 16:
        return False, f"ไฟล์ใหญ่เกินไป ({size_mb:.1f}MB) - สูงสุด 16MB"
    
    if size_mb < 0.001:
        return False, "ไฟล์ว่างเปล่า"
    
    # Check extension
    if not allowed_file(filepath):
        return False, "ประเภทไฟล์ไม่ถูกต้อง"
    
    return True, ""

def format_thai_date(dt: datetime = None) -> str:
    """
    Format datetime in Thai format
    
    Args:
        dt: Datetime object (default: now)
        
    Returns:
        str: Formatted Thai date string
    """
    if dt is None:
        dt = datetime.now()
    
    thai_months = [
        'มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน',
        'พฤษภาคม', 'มิถุนายน', 'กรกฎาคม', 'สิงหาคม',
        'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม'
    ]
    
    day = dt.day
    month = thai_months[dt.month - 1]
    year = dt.year + 543  # Convert to Buddhist Era
    time_str = dt.strftime('%H:%M:%S')
    
    return f"{day} {month} {year} เวลา {time_str}"
