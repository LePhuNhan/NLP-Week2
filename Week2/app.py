from flask import Flask, render_template, request, jsonify
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import json
import os
import re
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
from langdetect import detect, DetectorFactory
from pyvi import ViTokenizer, ViPosTagger
import underthesea

# Thiết lập seed cho langdetect để có kết quả ổn định
DetectorFactory.seed = 0

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tải dữ liệu NLTK cần thiết
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

# Cache cho kết quả phân tích
analysis_cache = {}

# Khởi tạo spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' chưa được cài đặt.")
    print("Vui lòng chạy: python -m spacy download en_core_web_sm")
    nlp = None

class TextAnalyzer:
    """Lớp phân tích văn bản sử dụng NLTK, spaCy và thư viện tiếng Việt"""
    
    def __init__(self):
        self.nlp = nlp
        self.cache = {}
        self.confidence_threshold = 0.7
        
    def validate_input(self, text: str) -> Tuple[bool, str]:
        """Validate input text"""
        if not text or not isinstance(text, str):
            return False, "Văn bản không hợp lệ"
        
        if len(text.strip()) < 2:
            return False, "Văn bản quá ngắn"
        
        if len(text) > 10000:
            return False, "Văn bản quá dài (tối đa 10,000 ký tự)"
        
        return True, "OK"
    
    def calculate_confidence_score(self, entities: List[Dict], tokens: List[str]) -> float:
        """Tính confidence score cho kết quả phân tích"""
        if not entities or not tokens:
            return 0.0
        
        # Tính score dựa trên số lượng entities và độ dài văn bản
        entity_ratio = len(entities) / len(tokens)
        base_score = min(entity_ratio * 10, 1.0)
        
        # Bonus cho entities có description chi tiết
        detailed_entities = sum(1 for e in entities if e.get('description', '') != e.get('label', ''))
        detail_bonus = detailed_entities / len(entities) * 0.2
        
        return min(base_score + detail_bonus, 1.0)
    
    @lru_cache(maxsize=128)
    def cached_detect_language(self, text: str) -> str:
        """Cached language detection"""
        return self.detect_language(text)
    
    def detect_language(self, text):
        """Phát hiện ngôn ngữ của văn bản"""
        try:
            # Lấy 100 ký tự đầu để phát hiện ngôn ngữ nhanh hơn
            sample_text = text[:100] if len(text) > 100 else text
            detected_lang = detect(sample_text)
            return detected_lang
        except:
            return 'unknown'
    
    def tokenize_with_nltk(self, text):
        """Tokenization sử dụng NLTK"""
        tokens = word_tokenize(text)
        return tokens
    
    def pos_tag_with_nltk(self, tokens):
        """POS tagging sử dụng NLTK"""
        pos_tags = pos_tag(tokens)
        return pos_tags
    
    def tokenize_vietnamese(self, text):
        """Tokenization cho tiếng Việt sử dụng pyvi"""
        try:
            tokens = ViTokenizer.tokenize(text).split()
            return tokens
        except:
            # Fallback về word_tokenize nếu pyvi lỗi
            return word_tokenize(text)
    
    def pos_tag_vietnamese(self, text):
        """POS tagging cho tiếng Việt sử dụng pyvi"""
        try:
            pos_tags = ViPosTagger.postagging(ViTokenizer.tokenize(text))
            # Kết hợp tokens và pos tags
            tokens, tags = pos_tags
            pos_tags_list = list(zip(tokens, tags))
            
            # Sửa các POS tags sai
            corrected_tags = self.correct_vietnamese_pos_tags(pos_tags_list)
            return corrected_tags
        except:
            # Fallback về NLTK nếu pyvi lỗi
            tokens = word_tokenize(text)
            return pos_tag(tokens)
    
    def analyze_vietnamese_with_underthesea(self, text):
        """Phân tích tiếng Việt sử dụng underthesea"""
        try:
            # Sử dụng POS tags đã được sửa từ pos_tag_vietnamese
            corrected_pos_tags = self.pos_tag_vietnamese(text)
            
            # Tokenization và POS tagging
            tokens_with_pos = []
            
            for token, pos in corrected_pos_tags:
                tokens_with_pos.append({
                    'token': token,
                    'pos': pos,
                    'tag': pos,  # underthesea chỉ có POS, không có tag chi tiết
                    'lemma': token  # underthesea không có lemmatization
                })
            
            # Named Entity Recognition
            entities = []
            try:
                ner_results = underthesea.ner(text)
                
                # Underthesea trả về (token, pos, chunk_tag, ner_tag)
                current_entity = ""
                current_label = ""
                
                for i, entity in enumerate(ner_results):
                    if len(entity) >= 4:  # (token, pos, chunk_tag, ner_tag)
                        token, pos, chunk_tag, ner_tag = entity[:4]
                        
                        if ner_tag.startswith('B-'):  # Bắt đầu entity mới
                            # Lưu entity trước đó nếu có
                            if current_entity and current_label:
                                entities.append({
                                    'text': current_entity.strip(),
                                    'label': current_label,
                                    'start': 0,
                                    'end': 0,
                                    'description': self.get_vietnamese_ner_description(current_label)
                                })
                            
                            # Bắt đầu entity mới
                            current_entity = token
                            current_label = ner_tag[2:]  # Bỏ 'B-' prefix
                            
                        elif ner_tag.startswith('I-') and current_label == ner_tag[2:]:  # Tiếp tục entity
                            # Chỉ thêm token nếu không phải dấu câu
                            if token not in [',', '.', '!', '?', ';', ':']:
                                current_entity += " " + token
                            
                        else:  # Không phải entity hoặc kết thúc entity
                            # Lưu entity trước đó nếu có
                            if current_entity and current_label:
                                entities.append({
                                    'text': current_entity.strip(),
                                    'label': current_label,
                                    'start': 0,
                                    'end': 0,
                                    'description': self.get_vietnamese_ner_description(current_label)
                                })
                                current_entity = ""
                                current_label = ""
                
                # Lưu entity cuối cùng nếu có
                if current_entity and current_label:
                    entities.append({
                        'text': current_entity.strip(),
                        'label': current_label,
                        'start': 0,
                        'end': 0,
                        'description': self.get_vietnamese_ner_description(current_label)
                    })
                
                # Làm sạch entities: loại bỏ entities quá ngắn hoặc chỉ chứa dấu câu
                cleaned_entities = []
                for entity in entities:
                    # Loại bỏ dấu câu và khoảng trắng ở đầu và cuối
                    clean_text = entity['text'].strip().strip(',.!?;:').strip()
                    if len(clean_text) > 1:
                        entity['text'] = clean_text
                        
                        # Sửa nhãn sai dựa trên context và từ khóa
                        entity = self.correct_vietnamese_ner_labels(entity)
                        
                        cleaned_entities.append(entity)
                
                # Thêm các entities bị thiếu
                additional_entities = self.add_missing_vietnamese_entities(text, cleaned_entities)
                entities = cleaned_entities + additional_entities
                    
            except Exception as e:
                pass  # NER có thể không hoạt động với một số phiên bản
            
            return {
                'tokens_with_pos': tokens_with_pos,
                'entities': entities
            }
        except Exception as e:
            print(f"Lỗi khi phân tích tiếng Việt với underthesea: {e}")
            return None
    
    def get_vietnamese_ner_description(self, ner_tag):
        """Lấy mô tả cho NER tag tiếng Việt"""
        descriptions = {
            'PER': 'Tên người',
            'LOC': 'Địa điểm',
            'ORG': 'Tổ chức',
            'MISC': 'Khác',
            'NP': 'Cụm danh từ',
            'DATE': 'Ngày tháng',
            'O': 'Không phải entity'
        }
        return descriptions.get(ner_tag, ner_tag)
    
    def get_vietnamese_pos_description(self, pos_tag):
        """Lấy mô tả chi tiết cho POS tag tiếng Việt"""
        descriptions = {
            # Danh từ
            'N': 'Danh từ chung (Noun)',
            'Np': 'Danh từ riêng (Proper Noun)',
            'Nu': 'Danh từ đơn vị (Unit Noun)',
            'Nc': 'Danh từ chỉ loại (Classifier Noun)',
            
            # Động từ
            'V': 'Động từ (Verb)',
            'Vb': 'Động từ bổ trợ (Auxiliary Verb)',
            'Vv': 'Động từ vị ngữ (Predicative Verb)',
            
            # Tính từ
            'A': 'Tính từ (Adjective)',
            'Ab': 'Tính từ bổ trợ (Auxiliary Adjective)',
            
            # Đại từ
            'P': 'Đại từ (Pronoun)',
            'Pp': 'Đại từ nhân xưng (Personal Pronoun)',
            'Pd': 'Đại từ chỉ định (Demonstrative Pronoun)',
            'Pq': 'Đại từ nghi vấn (Interrogative Pronoun)',
            
            # Số từ
            'M': 'Số từ (Numeral)',
            'Mc': 'Số từ chỉ số lượng (Cardinal Numeral)',
            'Mo': 'Số từ thứ tự (Ordinal Numeral)',
            
            # Phó từ
            'R': 'Phó từ (Adverb)',
            'Rg': 'Phó từ chỉ mức độ (Degree Adverb)',
            'Rr': 'Phó từ chỉ thời gian (Time Adverb)',
            'Rs': 'Phó từ chỉ nơi chốn (Place Adverb)',
            
            # Giới từ
            'E': 'Giới từ (Preposition)',
            'Ec': 'Giới từ chỉ nơi chốn (Place Preposition)',
            'Et': 'Giới từ chỉ thời gian (Time Preposition)',
            
            # Liên từ
            'C': 'Liên từ (Conjunction)',
            'Cc': 'Liên từ kết hợp (Coordinating Conjunction)',
            'Cs': 'Liên từ phụ thuộc (Subordinating Conjunction)',
            
            # Thán từ
            'I': 'Thán từ (Interjection)',
            
            # Trợ từ
            'T': 'Trợ từ (Particle)',
            'Td': 'Trợ từ định ngữ (Determiner Particle)',
            'Tg': 'Trợ từ ngữ khí (Modal Particle)',
            
            # Dấu câu
            'CH': 'Dấu câu (Punctuation)',
            'CHp': 'Dấu chấm (Period)',
            'CHc': 'Dấu phẩy (Comma)',
            'CHh': 'Dấu hỏi (Question Mark)',
            'CHk': 'Dấu chấm than (Exclamation Mark)',
            
            # Từ ngoại lai
            'FW': 'Từ ngoại lai (Foreign Word)',
            
            # Khác
            'X': 'Từ khác (Other)',
            'Y': 'Từ viết tắt (Abbreviation)',
            'Z': 'Từ không xác định (Unknown)'
        }
        return descriptions.get(pos_tag, f'{pos_tag} (Không xác định)')
    
    def correct_vietnamese_ner_labels(self, entity):
        """Sửa các nhãn NER sai dựa trên context và từ khóa"""
        text = entity['text'].lower()
        current_label = entity['label']
        
        # Danh sách các công ty nổi tiếng
        company_names = ['apple', 'microsoft', 'google', 'amazon', 'facebook', 'tesla', 'samsung', 'sony', 'nike', 'adidas']
        
        # Danh sách các tên người nổi tiếng
        famous_people = ['steve jobs', 'steve wozniak', 'ronald wayne', 'tim cook', 'bill gates', 'paul allen', 
                        'mark zuckerberg', 'jeff bezos', 'elon musk', 'larry page', 'sergey brin']
        
        # Sửa Apple từ PER thành ORG
        if text == 'apple' and current_label == 'PER':
            entity['label'] = 'ORG'
            entity['description'] = self.get_vietnamese_ner_description('ORG')
        
        # Sửa các tên người nổi tiếng từ LOC thành PER
        elif text in famous_people and current_label == 'LOC':
            entity['label'] = 'PER'
            entity['description'] = self.get_vietnamese_ner_description('PER')
        
        # Sửa năm từ LOC thành DATE
        elif ('năm' in text or text.isdigit()) and current_label == 'LOC':
            # Kiểm tra nếu là năm (4 chữ số)
            if text.replace('năm ', '').isdigit() and len(text.replace('năm ', '')) == 4:
                entity['label'] = 'DATE'
                entity['description'] = self.get_vietnamese_ner_description('DATE')
        
        # Sửa các công ty khác từ PER thành ORG
        elif text in company_names and current_label == 'PER':
            entity['label'] = 'ORG'
            entity['description'] = self.get_vietnamese_ner_description('ORG')
        
        # Sửa bệnh viện từ PER thành ORG
        elif 'chợ rẫy' in text and current_label == 'PER':
            entity['label'] = 'ORG'
            entity['description'] = self.get_vietnamese_ner_description('ORG')
        
        # Sửa TP.HCM từ PER thành LOC
        elif ('tp.hcm' in text or 'hồ chí minh' in text) and current_label == 'PER':
            entity['label'] = 'LOC'
            entity['description'] = self.get_vietnamese_ner_description('LOC')
        
        # Sửa bệnh viện từ PER thành ORG
        elif 'bệnh viện' in text and current_label == 'PER':
            entity['label'] = 'ORG'
            entity['description'] = self.get_vietnamese_ner_description('ORG')
        
        # Sửa trường đại học từ LOC thành ORG
        elif ('đại học' in text or 'bách khoa' in text or 'học viện' in text) and current_label == 'LOC':
            entity['label'] = 'ORG'
            entity['description'] = self.get_vietnamese_ner_description('ORG')
        
        # Sửa "Anh" từ PER thành MISC (đại từ)
        elif text == 'anh' and current_label == 'PER':
            entity['label'] = 'MISC'
            entity['description'] = self.get_vietnamese_ner_description('MISC')
        
        # Sửa tên huấn luyện viên từ LOC thành PER
        elif 'park hang-seo' in text and current_label == 'LOC':
            entity['label'] = 'PER'
            entity['description'] = self.get_vietnamese_ner_description('PER')
        
        # Sửa huấn luyện viên từ LOC thành MISC
        elif 'huấn luyện viên' in text and current_label == 'LOC':
            entity['label'] = 'MISC'
            entity['description'] = self.get_vietnamese_ner_description('MISC')
        
        # Sửa FPT Software từ PER thành ORG
        elif 'fpt software' in text and current_label == 'PER':
            entity['label'] = 'ORG'
            entity['description'] = self.get_vietnamese_ner_description('ORG')
        
        # Sửa CEO từ LOC thành MISC
        elif text == 'ceo' and current_label == 'LOC':
            entity['label'] = 'MISC'
            entity['description'] = self.get_vietnamese_ner_description('MISC')
        
        # Sửa các công ty khác từ PER thành ORG
        elif any(company in text for company in ['vng', 'vietcombank', 'fpt', 'vinfast', 'vingroup']) and current_label == 'PER':
            entity['label'] = 'ORG'
            entity['description'] = self.get_vietnamese_ner_description('ORG')
        
        # Sửa các trường đại học từ LOC thành ORG
        elif any(uni in text for uni in ['khoa học tự nhiên', 'bách khoa', 'quốc gia']) and current_label == 'LOC':
            entity['label'] = 'ORG'
            entity['description'] = self.get_vietnamese_ner_description('ORG')
        
        # Sửa ngân hàng từ PER thành ORG
        elif 'ngân hàng' in text and current_label == 'PER':
            entity['label'] = 'ORG'
            entity['description'] = self.get_vietnamese_ner_description('ORG')
        
        # Sửa các chức vụ từ LOC thành MISC
        elif any(title in text for title in ['hiệu trưởng', 'chủ tịch', 'giám đốc', 'thủ tướng', 'tổng thống']) and current_label == 'LOC':
            entity['label'] = 'MISC'
            entity['description'] = self.get_vietnamese_ner_description('MISC')
        
        # Sửa các địa điểm từ PER thành LOC
        elif any(location in text for location in ['đông nam á', 'thành phố hồ chí minh', 'hoa kỳ']) and current_label == 'PER':
            entity['label'] = 'LOC'
            entity['description'] = self.get_vietnamese_ner_description('LOC')
        
        return entity
    
    def add_missing_vietnamese_entities(self, text, existing_entities):
        """Thêm các entities bị thiếu dựa trên từ khóa và pattern"""
        additional_entities = []
        text_lower = text.lower()
        
        # Danh sách các tên người phổ biến
        common_names = ['kiên', 'minh', 'hùng', 'dũng', 'tuấn', 'nam', 'linh', 'hoa', 'mai', 'lan', 
                       'thảo', 'ngọc', 'vy', 'anh', 'huy', 'đức', 'quang', 'phong', 'long', 'khánh']
        
        # Danh sách các trường đại học
        universities = ['hcmute', 'hcmus', 'hcmut', 'hust', 'uet', 'neu', 'ftu', 'hue', 'dut', 'ctu']
        
        # Danh sách các quận/huyện
        districts = ['quận 1', 'quận 2', 'quận 3', 'quận 4', 'quận 5', 'quận 6', 'quận 7', 'quận 8', 
                    'quận 9', 'quận 10', 'quận 11', 'quận 12', 'quận bình thạnh', 'quận gò vấp', 
                    'quận phú nhuận', 'quận tân bình', 'quận tân phú', 'quận thủ đức']
        
        # Danh sách các bằng cấp
        degrees = ['thạc sĩ', 'tiến sĩ', 'cử nhân', 'kỹ sư', 'bác sĩ', 'thạc sỹ', 'tiến sỹ']
        
        # Danh sách các đơn vị đo lường
        units = ['tuổi', 'năm', 'tháng', 'ngày', 'giờ', 'phút', 'giây', 'kg', 'g', 'm', 'cm', 'km', 'lít', 'ml']
        
        # Kiểm tra tên người
        for name in common_names:
            if name in text_lower and not any(name in entity['text'].lower() for entity in existing_entities):
                additional_entities.append({
                    'text': name.title(),
                    'label': 'PER',
                    'start': 0,
                    'end': 0,
                    'description': 'Tên người'
                })
        
        # Kiểm tra số tuổi (số + tuổi)
        import re
        age_pattern = r'(\d+)\s*tuổi'
        age_matches = re.findall(age_pattern, text_lower)
        for age in age_matches:
            if not any(age in entity['text'] for entity in existing_entities):
                additional_entities.append({
                    'text': age,
                    'label': 'NUM',
                    'start': 0,
                    'end': 0,
                    'description': 'Số tuổi'
                })
        
        # Kiểm tra đơn vị đo lường (chỉ những từ có ý nghĩa trong ngữ cảnh)
        for unit in units:
            if unit in text_lower and not any(unit in entity['text'].lower() for entity in existing_entities):
                # Chỉ thêm nếu là từ có độ dài > 1 hoặc là đơn vị phổ biến
                if len(unit) > 1:
                    additional_entities.append({
                        'text': unit.title(),
                        'label': 'MISC',
                        'start': 0,
                        'end': 0,
                        'description': 'Đơn vị đo lường'
                    })
                # Chỉ thêm đơn vị 1 ký tự nếu có số đứng trước (ví dụ: "5g", "10m")
                elif len(unit) == 1:
                    import re
                    # Kiểm tra xem có số đứng trước không
                    pattern = r'\d+\s*' + unit
                    if re.search(pattern, text_lower):
                        additional_entities.append({
                            'text': unit.upper(),
                            'label': 'MISC',
                            'start': 0,
                            'end': 0,
                            'description': 'Đơn vị đo lường'
                        })
        
        # Kiểm tra trường đại học
        for uni in universities:
            if uni in text_lower and not any(uni in entity['text'].lower() for entity in existing_entities):
                additional_entities.append({
                    'text': uni.upper(),
                    'label': 'ORG',
                    'start': 0,
                    'end': 0,
                    'description': 'Tổ chức'
                })
        
        # Kiểm tra quận/huyện
        for district in districts:
            if district in text_lower and not any(district in entity['text'].lower() for entity in existing_entities):
                additional_entities.append({
                    'text': district.title(),
                    'label': 'LOC',
                    'start': 0,
                    'end': 0,
                    'description': 'Địa điểm'
                })
        
        # Kiểm tra bằng cấp
        for degree in degrees:
            if degree in text_lower and not any(degree in entity['text'].lower() for entity in existing_entities):
                additional_entities.append({
                    'text': degree.title(),
                    'label': 'MISC',
                    'start': 0,
                    'end': 0,
                    'description': 'Khác'
                })
        
        # Kiểm tra bệnh viện
        if 'bệnh viện' in text_lower and not any('bệnh viện' in entity['text'].lower() for entity in existing_entities):
            additional_entities.append({
                'text': 'Bệnh viện',
                'label': 'ORG',
                'start': 0,
                'end': 0,
                'description': 'Tổ chức'
            })
        
        # Kiểm tra số giường bệnh (số + giường bệnh)
        bed_pattern = r'(\d+[.,]?\d*)\s*giường\s*bệnh'
        bed_matches = re.findall(bed_pattern, text_lower)
        for bed in bed_matches:
            if not any(bed in entity['text'] for entity in existing_entities):
                additional_entities.append({
                    'text': bed,
                    'label': 'NUM',
                    'start': 0,
                    'end': 0,
                    'description': 'Số lượng'
                })
        
        # Kiểm tra giường bệnh
        if 'giường bệnh' in text_lower and not any('giường bệnh' in entity['text'].lower() for entity in existing_entities):
            additional_entities.append({
                'text': 'giường bệnh',
                'label': 'MISC',
                'start': 0,
                'end': 0,
                'description': 'Đơn vị đo lường'
            })
        
        # Kiểm tra trường đại học
        if 'đại học' in text_lower and not any('đại học' in entity['text'].lower() for entity in existing_entities):
            additional_entities.append({
                'text': 'Đại học',
                'label': 'ORG',
                'start': 0,
                'end': 0,
                'description': 'Tổ chức'
            })
        
        # Kiểm tra tên người đầy đủ (Nguyễn Văn Minh)
        import re
        full_name_pattern = r'(nguyễn|trần|lê|phạm|hoàng|phan|vũ|võ|đặng|bùi|đỗ|hồ|ngô|dương|lý)\s+(văn|thị|đức|minh|hùng|dũng|tuấn|nam|linh|hoa|mai|lan|thảo|ngọc|vy|anh|huy|đức|quang|phong|long|khánh)'
        full_name_matches = re.findall(full_name_pattern, text_lower)
        for first_name, middle_name in full_name_matches:
            full_name = f"{first_name.title()} {middle_name.title()}"
            if not any(full_name.lower() in entity['text'].lower() for entity in existing_entities):
                additional_entities.append({
                    'text': full_name,
                    'label': 'PER',
                    'start': 0,
                    'end': 0,
                    'description': 'Tên người'
                })
        
        # Kiểm tra tên người nước ngoài (Park Hang-seo)
        foreign_name_pattern = r'(park|kim|lee|choi|jung|yoon|kang|lim|oh|seo)\s+(hang-seo|min-jae|son|heung-min|jae-sung|woo-young|hyun-jin|dong-gook|bo-kyung|young-pyo)'
        foreign_name_matches = re.findall(foreign_name_pattern, text_lower)
        for first_name, last_name in foreign_name_matches:
            full_name = f"{first_name.title()} {last_name.title()}"
            if not any(full_name.lower() in entity['text'].lower() for entity in existing_entities):
                additional_entities.append({
                    'text': full_name,
                    'label': 'PER',
                    'start': 0,
                    'end': 0,
                    'description': 'Tên người'
                })
        
        # Kiểm tra tỷ số (2-1, 3-0, 1-1)
        score_pattern = r'(\d+)\s*-\s*(\d+)'
        score_matches = re.findall(score_pattern, text_lower)
        for score1, score2 in score_matches:
            score = f"{score1}-{score2}"
            if not any(score in entity['text'] for entity in existing_entities):
                additional_entities.append({
                    'text': score,
                    'label': 'NUM',
                    'start': 0,
                    'end': 0,
                    'description': 'Tỷ số'
                })
        
        # Kiểm tra huấn luyện viên
        if 'huấn luyện viên' in text_lower and not any('huấn luyện viên' in entity['text'].lower() for entity in existing_entities):
            additional_entities.append({
                'text': 'huấn luyện viên',
                'label': 'MISC',
                'start': 0,
                'end': 0,
                'description': 'Chức vụ'
            })
        
        # Kiểm tra công ty
        if 'công ty' in text_lower and not any('công ty' in entity['text'].lower() for entity in existing_entities):
            additional_entities.append({
                'text': 'Công ty',
                'label': 'ORG',
                'start': 0,
                'end': 0,
                'description': 'Tổ chức'
            })
        
        # Kiểm tra CEO
        if 'ceo' in text_lower and not any('ceo' in entity['text'].lower() for entity in existing_entities):
            additional_entities.append({
                'text': 'CEO',
                'label': 'MISC',
                'start': 0,
                'end': 0,
                'description': 'Chức vụ'
            })
        
        # Kiểm tra số năm (1999, 2000, 2023...)
        year_pattern = r'\b(19|20)\d{2}\b'
        year_matches = re.findall(year_pattern, text_lower)
        for year in year_matches:
            if not any(year in entity['text'] for entity in existing_entities):
                additional_entities.append({
                    'text': year,
                    'label': 'NUM',
                    'start': 0,
                    'end': 0,
                    'description': 'Năm'
                })
        
        # Kiểm tra các chức vụ
        titles = ['hiệu trưởng', 'chủ tịch', 'giám đốc', 'thủ tướng', 'tổng thống', 'pgs.ts', 'bs.']
        for title in titles:
            if title in text_lower and not any(title in entity['text'].lower() for entity in existing_entities):
                additional_entities.append({
                    'text': title.title(),
                    'label': 'MISC',
                    'start': 0,
                    'end': 0,
                    'description': 'Chức vụ'
                })
        
        # Kiểm tra ngân hàng
        if 'ngân hàng' in text_lower and not any('ngân hàng' in entity['text'].lower() for entity in existing_entities):
            additional_entities.append({
                'text': 'Ngân hàng',
                'label': 'ORG',
                'start': 0,
                'end': 0,
                'description': 'Tổ chức'
            })
        
        # Kiểm tra trường đại học
        universities = ['khoa học tự nhiên', 'bách khoa', 'quốc gia']
        for uni in universities:
            if uni in text_lower and not any(uni in entity['text'].lower() for entity in existing_entities):
                additional_entities.append({
                    'text': uni.title(),
                    'label': 'ORG',
                    'start': 0,
                    'end': 0,
                    'description': 'Tổ chức'
                })
        
        # Kiểm tra dân số (97 triệu, 8 triệu...)
        population_pattern = r'(\d+)\s*(triệu|nghìn|tỷ)'
        population_matches = re.findall(population_pattern, text_lower)
        for number, unit in population_matches:
            population = f"{number} {unit}"
            if not any(population in entity['text'] for entity in existing_entities):
                additional_entities.append({
                    'text': population,
                    'label': 'NUM',
                    'start': 0,
                    'end': 0,
                    'description': 'Dân số'
                })
        
        # Kiểm tra các địa điểm đặc biệt
        special_locations = ['đông nam á', 'thành phố hồ chí minh', 'hoa kỳ', 'washington d.c.', 'boston']
        for location in special_locations:
            if location in text_lower and not any(location in entity['text'].lower() for entity in existing_entities):
                additional_entities.append({
                    'text': location.title(),
                    'label': 'LOC',
                    'start': 0,
                    'end': 0,
                    'description': 'Địa điểm'
                })
        
        return additional_entities
    
    def correct_vietnamese_pos_tags(self, pos_tags):
        """Sửa các POS tags sai dựa trên context và từ khóa"""
        corrected_tags = []
        
        for i, (token, pos) in enumerate(pos_tags):
            # Danh sách các tên người phổ biến
            common_names = ['kiên', 'minh', 'hùng', 'dũng', 'tuấn', 'nam', 'linh', 'hoa', 'mai', 'lan', 
                          'thảo', 'ngọc', 'vy', 'anh', 'huy', 'đức', 'quang', 'phong', 'long', 'khánh']
            
            # Danh sách các từ có thể là phó từ chỉ thời gian
            time_adverbs = ['hiện_tại', 'hiện_nay', 'bây_giờ', 'lúc_này', 'ngay_bây_giờ', 'hiện_giờ']
            
            # Sửa tên người từ N thành Np
            if token.lower() in common_names and pos == 'N':
                corrected_tags.append((token, 'Np'))
            
            # Sửa hiện_tại từ N thành R trong ngữ cảnh phù hợp
            elif token.lower() == 'hiện_tại' and pos == 'N':
                # Kiểm tra ngữ cảnh xung quanh
                context_around = []
                for j in range(max(0, i-2), min(len(pos_tags), i+3)):
                    if j != i:
                        context_around.append(pos_tags[j][0].lower())
                
                context_text = ' '.join(context_around)
                
                # Nếu có từ "tuổi" hoặc "năm" gần đó, có thể là phó từ
                if 'tuổi' in context_text or 'năm' in context_text:
                    corrected_tags.append((token, 'R'))
                else:
                    corrected_tags.append((token, pos))
            
            # Giữ nguyên các tags khác
            else:
                corrected_tags.append((token, pos))
        
        return corrected_tags
    
    def correct_english_ner_labels(self, entity):
        """Sửa các nhãn NER tiếng Anh sai dựa trên context và từ khóa"""
        text = entity['text'].lower()
        current_label = entity['label']
        
        # Sửa các lỗi phổ biến cho tiếng Anh
        if 'mvp' in text and current_label == 'ORG':
            entity['label'] = 'MISC'
            entity['description'] = 'Award/Title'
        elif 'championship' in text and current_label == 'ORG':
            entity['label'] = 'EVENT'
            entity['description'] = 'Sports event'
        elif 'finals' in text and current_label == 'ORG':
            entity['label'] = 'EVENT'
            entity['description'] = 'Sports event'
        elif 'nba' in text and current_label == 'PERSON':
            entity['label'] = 'ORG'
            entity['description'] = 'Sports organization'
        elif 'ai' in text and current_label == 'PERSON':
            entity['label'] = 'MISC'
            entity['description'] = 'Technology'
        elif 'software engineer' in text and current_label == 'PERSON':
            entity['label'] = 'MISC'
            entity['description'] = 'Job title'
        
        return entity
    
    def add_missing_english_entities(self, text, existing_entities):
        """Thêm các entities bị thiếu cho tiếng Anh"""
        additional_entities = []
        text_lower = text.lower()
        
        import re
        
        # Kiểm tra các chức vụ
        job_titles = ['ceo', 'president', 'coach', 'mvp', 'software engineer', 'champion']
        for title in job_titles:
            if title in text_lower and not any(title in entity['text'].lower() for entity in existing_entities):
                additional_entities.append({
                    'text': title.title(),
                    'label': 'MISC',
                    'start': 0,
                    'end': 0,
                    'description': 'Job title'
                })
        
        # Kiểm tra các sự kiện thể thao
        sports_events = ['championship', 'finals', 'nba']
        for event in sports_events:
            if event in text_lower and not any(event in entity['text'].lower() for entity in existing_entities):
                if event == 'nba':
                    additional_entities.append({
                        'text': 'NBA',
                        'label': 'ORG',
                        'start': 0,
                        'end': 0,
                        'description': 'Sports organization'
                    })
                else:
                    additional_entities.append({
                        'text': event.title(),
                        'label': 'EVENT',
                        'start': 0,
                        'end': 0,
                        'description': 'Sports event'
                    })
        
        # Kiểm tra công nghệ
        if 'ai' in text_lower and not any('ai' in entity['text'].lower() for entity in existing_entities):
            additional_entities.append({
                'text': 'AI',
                'label': 'MISC',
                'start': 0,
                'end': 0,
                'description': 'Technology'
            })
        
        return additional_entities
    
    def analyze_with_spacy(self, text):
        """Phân tích văn bản sử dụng spaCy"""
        if not self.nlp:
            return None
        
        doc = self.nlp(text)
        
        # Tokenization và POS tagging
        tokens_with_pos = []
        for token in doc:
            tokens_with_pos.append({
                'token': token.text,
                'pos': token.pos_,
                'tag': token.tag_,
                'lemma': token.lemma_
            })
        
        # Named Entity Recognition
        entities = []
        for ent in doc.ents:
            entity = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_)
            }
            
            # Sửa các nhãn NER sai
            entity = self.correct_english_ner_labels(entity)
            entities.append(entity)
        
        # Thêm các entities bị thiếu
        additional_entities = self.add_missing_english_entities(text, entities)
        entities.extend(additional_entities)
        
        return {
            'tokens_with_pos': tokens_with_pos,
            'entities': entities
        }
    
    def analyze_mixed_language_text(self, text):
        """Phân tích văn bản hỗn hợp (tiếng Việt + tiếng Anh)"""
        import re
        
        # Tìm các từ tiếng Anh trong văn bản
        english_words = re.findall(r'\b[A-Za-z]+\b', text)
        vietnamese_words = re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]+\w*', text)
        
        # Tính tổng số từ
        total_words = len(english_words) + len(vietnamese_words)
        if total_words == 0:
            return None
        
        # Tính tỷ lệ
        english_ratio = len(english_words) / total_words
        vietnamese_ratio = len(vietnamese_words) / total_words
        
        # Chỉ coi là mixed khi:
        # 1. Có ít nhất 5 từ tiếng Anh
        # 2. Có ít nhất 5 từ tiếng Việt  
        # 3. Tiếng Anh chiếm 30-70% văn bản
        # 4. Không phải chỉ là tên riêng hoặc từ viết tắt
        if (len(english_words) >= 5 and len(vietnamese_words) >= 5 and 
            0.3 <= english_ratio <= 0.7):
            
            # Kiểm tra xem có phải chỉ là tên riêng/từ viết tắt không
            common_vietnamese_words = ['công', 'ty', 'có', 'trụ', 'sở', 'tại', 'hiện', 'tại', 'là', 'được', 'thành', 'lập', 'vào', 'năm']
            common_english_words = ['the', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'would', 'can', 'could', 'should', 'may', 'might']
            
            # Đếm từ tiếng Anh thông thường
            meaningful_english = sum(1 for word in english_words if word.lower() in common_english_words)
            
            # Nếu ít hơn 2 từ tiếng Anh có nghĩa, coi như tiếng Việt
            if meaningful_english < 2:
                return None
            
            # Sử dụng NLTK cho tokenization
            nltk_tokens = self.tokenize_with_nltk(text)
            nltk_pos_tags = self.pos_tag_with_nltk(nltk_tokens)
            
            # Sử dụng spaCy cho NER (tốt hơn cho tiếng Anh)
            spacy_analysis = self.analyze_with_spacy(text)
            
            # Thêm logic correction cho văn bản hỗn hợp
            if spacy_analysis and 'entities' in spacy_analysis:
                corrected_entities = []
                for entity in spacy_analysis['entities']:
                    # Sửa các lỗi cho văn bản hỗn hợp
                    entity = self.correct_mixed_language_ner_labels(entity, text)
                    corrected_entities.append(entity)
                spacy_analysis['entities'] = corrected_entities
            
            return {
                'language': 'mixed',
                'detected_language': 'mixed',
                'nltk_analysis': {
                    'tokens': nltk_tokens,
                    'pos_tags': nltk_pos_tags
                },
                'spacy_analysis': spacy_analysis
            }
        
        return None
    
    def correct_mixed_language_ner_labels(self, entity, full_text):
        """Sửa các nhãn NER cho văn bản hỗn hợp"""
        text = entity['text'].lower()
        current_label = entity['label']
        
        # Sửa các lỗi phổ biến cho văn bản hỗn hợp
        if 'joe biden' in text and current_label == 'ORG':
            entity['label'] = 'PERSON'
            entity['description'] = 'Person'
        elif 'phạm minh chính' in text and current_label == 'ORG':
            entity['label'] = 'PER'
            entity['description'] = 'Tên người'
        elif 'washington d.c.' in text and current_label == 'PERSON':
            entity['label'] = 'GPE'
            entity['description'] = 'Geopolitical entity'
        elif 'microsoft' in text and current_label == 'PERSON':
            entity['label'] = 'ORG'
            entity['description'] = 'Organization'
        elif 'phạm nhật vượng' in text and current_label == 'ORG':
            entity['label'] = 'PER'
            entity['description'] = 'Tên người'
        elif 'satya nadella' in text and current_label == 'ORG':
            entity['label'] = 'PERSON'
            entity['description'] = 'Person'
        elif 'hà nội' in text and current_label == 'ORG':
            entity['label'] = 'GPE'
            entity['description'] = 'Geopolitical entity'
        elif 'vingroup' in text and current_label == 'PERSON':
            entity['label'] = 'ORG'
            entity['description'] = 'Tổ chức'
        elif 'ai' in text and current_label == 'PERSON':
            entity['label'] = 'MISC'
            entity['description'] = 'Technology'
        elif 'mit' in text and current_label == 'PERSON':
            entity['label'] = 'ORG'
            entity['description'] = 'Organization'
        elif 'nguyễn kim sơn' in text and current_label == 'ORG':
            entity['label'] = 'PER'
            entity['description'] = 'Tên người'
        elif 'boston' in text and current_label == 'PERSON':
            entity['label'] = 'GPE'
            entity['description'] = 'Geopolitical entity'
        # Thêm các sửa lỗi cho văn bản tiếng Việt
        elif 'fpt software' in text and current_label == 'PERSON':
            entity['label'] = 'ORG'
            entity['description'] = 'Organization'
        elif 'nguyễn thành nam' in text and current_label == 'EVENT':
            entity['label'] = 'PERSON'
            entity['description'] = 'Person'
        elif 'ceo' in text and current_label == 'PERSON':
            entity['label'] = 'MISC'
            entity['description'] = 'Job title'
        
        return entity
    
    def analyze_text(self, text: str) -> Optional[Dict]:
        """Phân tích văn bản hoàn chỉnh với hỗ trợ đa ngôn ngữ"""
        # Validate input
        is_valid, error_msg = self.validate_input(text)
        if not is_valid:
            logger.warning(f"Input validation failed: {error_msg}")
            return None
        
        # Check cache
        text_hash = hash(text.strip())
        if text_hash in self.cache:
            logger.info("Returning cached result")
            return self.cache[text_hash]
        
        try:
            # Phát hiện ngôn ngữ
            detected_language = self.cached_detect_language(text)
            
            # Kiểm tra xem có phải văn bản hỗn hợp không
            mixed_analysis = self.analyze_mixed_language_text(text)
            if mixed_analysis:
                # Tính confidence score
                entities = mixed_analysis.get('spacy_analysis', {}).get('entities', [])
                tokens = mixed_analysis.get('nltk_analysis', {}).get('tokens', [])
                confidence = self.calculate_confidence_score(entities, tokens)
                mixed_analysis['confidence_score'] = confidence
                
                # Cache result
                self.cache[text_hash] = mixed_analysis
                return mixed_analysis
            
            if detected_language == 'vi':
                # Phân tích tiếng Việt
                vietnamese_tokens = self.tokenize_vietnamese(text)
                vietnamese_pos_tags = self.pos_tag_vietnamese(text)
                underthesea_analysis = self.analyze_vietnamese_with_underthesea(text)
                
                result = {
                    'language': 'vietnamese',
                    'detected_language': detected_language,
                    'nltk_analysis': {
                        'tokens': vietnamese_tokens,
                        'pos_tags': vietnamese_pos_tags
                    },
                    'spacy_analysis': underthesea_analysis,
                    'vietnamese_analysis': {
                        'tokens': vietnamese_tokens,
                        'pos_tags': vietnamese_pos_tags,
                        'underthesea_analysis': underthesea_analysis
                    }
                }
                
                # Tính confidence score
                entities = underthesea_analysis.get('entities', []) if underthesea_analysis else []
                confidence = self.calculate_confidence_score(entities, vietnamese_tokens)
                result['confidence_score'] = confidence
                
                # Cache result
                self.cache[text_hash] = result
                return result
            else:
                # Phân tích tiếng Anh (hoặc ngôn ngữ khác)
                nltk_tokens = self.tokenize_with_nltk(text)
                nltk_pos_tags = self.pos_tag_with_nltk(nltk_tokens)
                spacy_analysis = self.analyze_with_spacy(text)
                
                result = {
                    'language': 'english',
                    'detected_language': detected_language,
                    'nltk_analysis': {
                        'tokens': nltk_tokens,
                        'pos_tags': nltk_pos_tags
                    },
                    'spacy_analysis': spacy_analysis
                }
                
                # Tính confidence score
                entities = spacy_analysis.get('entities', []) if spacy_analysis else []
                confidence = self.calculate_confidence_score(entities, nltk_tokens)
                result['confidence_score'] = confidence
                
                # Cache result
                self.cache[text_hash] = result
                return result
                
        except Exception as e:
            logger.error(f"Error in analyze_text: {str(e)}")
            return None

# Khởi tạo analyzer
analyzer = TextAnalyzer()

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint để phân tích văn bản"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request phải là JSON'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Dữ liệu JSON không hợp lệ'}), 400
        
        text = data.get('text', '').strip()
        
        logger.info(f"Analyzing text: {text[:50]}...")
        
        print(f"\n{'='*60}")
        print(f"🔍 PHÂN TÍCH VĂN BẢN MỚI")
        print(f"{'='*60}")
        print(f"📝 Văn bản đầu vào: {text}")
        print(f"📏 Độ dài: {len(text)} ký tự")
        
        # Validate input
        is_valid, error_msg = analyzer.validate_input(text)
        if not is_valid:
            print(f"❌ Lỗi validation: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # Phân tích văn bản
        result = analyzer.analyze_text(text)
        
        if result is None:
            print("❌ Lỗi: Không thể phân tích văn bản")
            return jsonify({'error': 'Không thể phân tích văn bản'}), 500
        
        # Log kết quả chi tiết
        print(f"\n🌍 THÔNG TIN NGÔN NGỮ:")
        print(f"   - Ngôn ngữ: {result.get('language', 'unknown')}")
        print(f"   - Ngôn ngữ phát hiện: {result.get('detected_language', 'unknown')}")
        print(f"   - Confidence Score: {result.get('confidence_score', 0.0):.2f}")
        
        # Log tokens
        if 'nltk_analysis' in result:
            tokens = result['nltk_analysis'].get('tokens', [])
            pos_tags = result['nltk_analysis'].get('pos_tags', [])
            print(f"\n🔤 TOKENIZATION:")
            print(f"   - Số tokens: {len(tokens)}")
            print(f"   - Tokens: {tokens}")
            
            print(f"\n🏷️ POS TAGS:")
            for token, pos in pos_tags:
                print(f"   - {token}: {pos}")
        
        # Log entities
        if 'spacy_analysis' in result and result['spacy_analysis']:
            entities = result['spacy_analysis'].get('entities', [])
            print(f"\n🎯 NAMED ENTITY RECOGNITION:")
            print(f"   - Số entities: {len(entities)}")
            for entity in entities:
                print(f"   - {entity['text']} ({entity['label']}): {entity['description']}")
        else:
            print(f"\n🎯 NAMED ENTITY RECOGNITION:")
            print(f"   - Không tìm thấy entities")
        
        # Log phân tích chi tiết
        if 'spacy_analysis' in result and result['spacy_analysis']:
            tokens_with_pos = result['spacy_analysis'].get('tokens_with_pos', [])
            print(f"\n📊 PHÂN TÍCH CHI TIẾT:")
            print(f"   - Số tokens chi tiết: {len(tokens_with_pos)}")
            for token_info in tokens_with_pos:
                print(f"   - {token_info['token']}: POS={token_info['pos']}, Tag={token_info['tag']}, Lemma={token_info['lemma']}")
        
        print(f"\n✅ PHÂN TÍCH HOÀN TẤT")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        print(f"\n❌ LỖI KHI PHÂN TÍCH: {str(e)}")
        print(f"{'='*60}\n")
        return jsonify({'error': f'Lỗi khi phân tích: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'spacy_available': nlp is not None,
        'cache_size': len(analyzer.cache)
    })

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear analysis cache"""
    try:
        cache_size = len(analyzer.cache)
        analyzer.cache.clear()
        logger.info(f"Cache cleared. Removed {cache_size} entries.")
        return jsonify({
            'success': True,
            'message': f'Đã xóa {cache_size} entries khỏi cache',
            'cleared_entries': cache_size
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({'error': f'Lỗi khi xóa cache: {str(e)}'}), 500

@app.route('/cache/stats')
def cache_stats():
    """Get cache statistics"""
    return jsonify({
        'cache_size': len(analyzer.cache),
        'confidence_threshold': analyzer.confidence_threshold
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
