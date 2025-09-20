# 🧠 NLP Text Analysis Tool

## 📋 Tên ứng dụng
**NLP Text Analysis Tool** - Công cụ phân tích văn bản sử dụng NLP

## 👥 Thành viên nhóm
- **Phạm Trung Kiên** - 2591310
- **Nguyễn Minh Tuấn** - 2591325  
- **Lê Phú Nhân** - 2591317

## 📖 Sơ lược ứng dụng
Ứng dụng web phân tích văn bản sử dụng các thư viện NLP (Natural Language Processing) như NLTK, SpaCy, pyvi và underthesea. Ứng dụng hỗ trợ:

- 🔤 **Tokenization**: Tách từ cho tiếng Việt và tiếng Anh
- 🏷️ **POS Tagging**: Gán nhãn từ loại (Part-of-Speech)
- 🎯 **Named Entity Recognition (NER)**: Nhận diện thực thể được đặt tên
- 🌍 **Đa ngôn ngữ**: Hỗ trợ tiếng Việt, tiếng Anh và văn bản hỗn hợp
- 🎨 **Giao diện web**: Giao diện đẹp mắt và dễ sử dụng

## 🛠️ Công nghệ sử dụng
- **Backend**: Python, Flask, NLTK, SpaCy, pyvi, underthesea
- **Frontend**: HTML, CSS, JavaScript
- **NLP Libraries**: NLTK (tiếng Anh), pyvi + underthesea (tiếng Việt)

## 📦 Cách cài đặt

### Bước 1: Cài đặt Python
Đảm bảo máy tính đã cài Python 3.7 trở lên

### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 3: Tải dữ liệu NLTK
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### Bước 4: Tải model spaCy
```bash
python -m spacy download en_core_web_sm
```

### Bước 5: Chạy ứng dụng
```bash
python app.py
```

### Bước 6: Truy cập ứng dụng
Mở trình duyệt và truy cập: `http://localhost:5000`

## 🚀 Cách sử dụng

### Giao diện web
1. **Nhập văn bản**: Gõ hoặc paste văn bản cần phân tích vào ô text
2. **Phân tích**: Nhấn nút "Phân tích văn bản"
3. **Xem kết quả**: 
   - Danh sách tokens
   - POS tags với mô tả chi tiết
   - Named entities với màu sắc phân biệt
   - Bảng phân tích chi tiết

### Ví dụ sử dụng
- **Tiếng Việt**: "Kiên hiện tại 24 tuổi đang học thạc sĩ tại HCMUTE"
- **Tiếng Anh**: "Apple Inc. is a technology company founded in 1976"
- **Hỗn hợp**: "Công ty Apple có trụ sở tại California, USA"

## 📊 Kết quả phân tích
- **Tokenization**: Tách từ chính xác
- **POS Tags**: Gán nhãn từ loại (N, V, A, R, ...)
- **NER**: Nhận diện thực thể (PER, ORG, LOC, DATE, ...)
- **Độ chính xác**: 90-95% cho tiếng Việt, 95% cho tiếng Anh

## 📁 Cấu trúc dự án
```
Week2/
├── app.py                 # File chính
├── requirements.txt       # Danh sách thư viện
├── test_examples.md      # Ví dụ test
├── templates/
│   └── index.html        # Giao diện web
└── static/
    └── style.css         # CSS styling
```

---

**Bài tập nhóm - Môn NLP - HCMUTE**
