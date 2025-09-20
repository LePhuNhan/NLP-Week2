"""
Script thiết lập môi trường cho ứng dụng NLP
Chạy script này để cài đặt tất cả dependencies và models cần thiết
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Chạy command và hiển thị kết quả"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} thành công!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} thất bại!")
        print(f"Lỗi: {e.stderr}")
        return False

def main():
    print("🚀 Thiết lập môi trường cho ứng dụng NLP")
    print("=" * 50)
    
    # Kiểm tra Python version
    if sys.version_info < (3, 7):
        print("❌ Cần Python 3.7 trở lên!")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Cài đặt requirements
    if not run_command("pip install -r requirements.txt", "Cài đặt Python packages"):
        print("❌ Không thể cài đặt requirements. Vui lòng kiểm tra lại.")
        sys.exit(1)
    
    # Tải spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Tải spaCy English model"):
        print("⚠️  Không thể tải spaCy model. Ứng dụng vẫn có thể chạy với NLTK.")
    
    # Tải NLTK data
    print("\n🔄 Tải NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✅ NLTK data đã được tải!")
    except Exception as e:
        print(f"⚠️  Lỗi khi tải NLTK data: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Thiết lập hoàn tất!")
    print("\n📋 Hướng dẫn chạy ứng dụng:")
    print("1. Chạy: python app.py")
    print("2. Mở trình duyệt: http://localhost:5000")
    print("3. Nhập văn bản và phân tích!")
    
    print("\n💡 Ví dụ văn bản để test:")
    print("Apple Inc. is located in Cupertino, California. Tim Cook is the CEO.")

if __name__ == "__main__":
    main()
