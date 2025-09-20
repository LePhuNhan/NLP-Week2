# Công cụ Phân tích Văn bản NLP - Hỗ trợ Tiếng Việt 🇻🇳

Ứng dụng web phân tích văn bản với hỗ trợ **Tiếng Việt** và **Tiếng Anh**, sử dụng các thư viện NLP chuyên dụng cho từng ngôn ngữ.

## ✨ Tính năng mới - Hỗ trợ Tiếng Việt

- **🌍 Phát hiện ngôn ngữ tự động**: Tự động nhận diện tiếng Việt và tiếng Anh
- **🇻🇳 Xử lý tiếng Việt**: Sử dụng pyvi và underthesea cho tokenization, POS tagging, NER
- **🇺🇸 Xử lý tiếng Anh**: Sử dụng NLTK và spaCy như trước
- **🎯 Named Entity Recognition**: Nhận diện tên người, địa điểm, tổ chức trong tiếng Việt
- **📊 Giao diện thông minh**: Hiển thị thông tin ngôn ngữ được phát hiện

## 🚀 Cài đặt

### 1. Clone và di chuyển vào thư mục
```bash
cd Week2
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Tải spaCy model (cho tiếng Anh)
```bash
python -m spacy download en_core_web_sm
```

## 🎯 Sử dụng

### Chạy ứng dụng
```bash
python app.py
```

Truy cập: `http://localhost:5000`

## 📝 Ví dụ sử dụng

### Ví dụ 1: Văn bản tiếng Việt
```
Công ty Apple Inc. có trụ sở tại Cupertino, California. Tim Cook là CEO của công ty. Công ty được thành lập vào năm 1976 bởi Steve Jobs, Steve Wozniak và Ronald Wayne.
```

**Kết quả mong đợi:**
- **Ngôn ngữ**: Tiếng Việt 🇻🇳 (vi)
- **Tokens**: Công_ty, Apple, Inc., có, trụ_sở, tại, Cupertino, California, Tim, Cook, là, CEO, của, công_ty
- **POS Tags**: Công_ty (N), Apple (Np), Inc. (Np), có (V), trụ_sở (N), tại (E), Cupertino (Np), California (Np)
- **Entities**: Apple Inc. (ORG), Cupertino (LOC), California (LOC), Tim Cook (PER), Steve Jobs (PER), 1976 (DATE)

### Ví dụ 2: Văn bản tiếng Anh
```
Apple Inc. is located in Cupertino, California. Tim Cook is the CEO of the company. The company was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.
```

**Kết quả mong đợi:**
- **Ngôn ngữ**: English 🇺🇸 (en)
- **Tokens**: Apple, Inc., is, located, in, Cupertino, California, Tim, Cook, is, the, CEO
- **POS Tags**: Apple (NNP), Inc. (NNP), is (VBZ), located (VBN), in (IN), Cupertino (NNP)
- **Entities**: Apple Inc. (ORG), Cupertino (GPE), California (GPE), Tim Cook (PERSON), Steve Jobs (PERSON), 1976 (DATE)

### Ví dụ 3: Văn bản về Việt Nam
```
Hà Nội là thủ đô của Việt Nam. Chủ tịch nước hiện tại là Nguyễn Xuân Phúc. Thành phố Hồ Chí Minh là thành phố lớn nhất của đất nước.
```

**Kết quả mong đợi:**
- **Ngôn ngữ**: Tiếng Việt 🇻🇳 (vi)
- **Entities**: Hà Nội (LOC), Việt Nam (LOC), Nguyễn Xuân Phúc (PER), Thành phố Hồ Chí Minh (LOC)

## 🔧 Công nghệ sử dụng

### Tiếng Việt
- **pyvi**: Tokenization và POS tagging cho tiếng Việt
- **underthesea**: Phân tích ngôn ngữ tự nhiên tiếng Việt
- **langdetect**: Phát hiện ngôn ngữ

### Tiếng Anh
- **NLTK**: Tokenization và POS tagging
- **spaCy**: Phân tích ngôn ngữ tự nhiên nâng cao

### Backend & Frontend
- **Flask**: Web framework
- **HTML5, CSS3, JavaScript**: Giao diện người dùng

## 📊 So sánh kết quả

| Tính năng | Tiếng Việt | Tiếng Anh |
|-----------|------------|-----------|
| Tokenization | pyvi | NLTK |
| POS Tagging | pyvi | NLTK |
| NER | underthesea | spaCy |
| Lemmatization | ❌ | ✅ |
| Dependency Parsing | ❌ | ✅ |

## 🎨 Giao diện

- **Thông tin ngôn ngữ**: Hiển thị cờ quốc gia và tên ngôn ngữ được phát hiện
- **Màu sắc phân biệt**: Các entity được đánh dấu màu khác nhau
- **Responsive design**: Hoạt động tốt trên mobile và desktop
- **Loading indicator**: Hiển thị trạng thái xử lý

## 🐛 Xử lý lỗi

- **Fallback mechanism**: Nếu thư viện tiếng Việt lỗi, sẽ dùng NLTK
- **Error handling**: Thông báo lỗi rõ ràng cho người dùng
- **Language detection failure**: Mặc định xử lý như tiếng Anh

## 🚀 Phát triển

### Thêm ngôn ngữ mới
1. Cài đặt thư viện NLP cho ngôn ngữ đó
2. Thêm logic phát hiện ngôn ngữ trong `detect_language()`
3. Tạo phương thức phân tích riêng trong `TextAnalyzer`
4. Cập nhật `analyze_text()` để xử lý ngôn ngữ mới

### Cải thiện hiệu suất
- Cache models sau lần load đầu tiên
- Xử lý bất đồng bộ cho văn bản dài
- Tối ưu hóa phát hiện ngôn ngữ

## 📈 Điểm mạnh

✅ **Hỗ trợ đa ngôn ngữ**: Tiếng Việt và tiếng Anh  
✅ **Phát hiện ngôn ngữ tự động**: Không cần chọn ngôn ngữ thủ công  
✅ **Thư viện chuyên dụng**: Sử dụng công cụ tốt nhất cho từng ngôn ngữ  
✅ **Giao diện thân thiện**: Hiển thị thông tin ngôn ngữ rõ ràng  
✅ **Fallback mechanism**: Đảm bảo ứng dụng luôn hoạt động  

## 🎯 Mục tiêu đạt được

- ✅ Tokenization cho tiếng Việt
- ✅ POS Tagging cho tiếng Việt  
- ✅ Named Entity Recognition cho tiếng Việt
- ✅ Phát hiện ngôn ngữ tự động
- ✅ Giao diện hỗ trợ đa ngôn ngữ
- ✅ **Thêm điểm cho khả năng xử lý tiếng Việt!** 🎉

## 📞 Hỗ trợ

Nếu gặp vấn đề với xử lý tiếng Việt, hãy kiểm tra:
1. Đã cài đặt đầy đủ `pyvi` và `underthesea`
2. Văn bản tiếng Việt có dấu đầy đủ
3. Kết nối internet để tải models (nếu cần)

---

**Chúc bạn sử dụng ứng dụng hiệu quả và đạt điểm cao!** 🌟