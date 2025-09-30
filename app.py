from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import datetime
import re
import tempfile
import warnings
warnings.filterwarnings("ignore")

# استبدال المكتبات الثقيلة بمكتبات أخف
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

try:
    from googletrans import Translator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

# إزالة المكتبات الثقيلة
HAS_SENTENCE_TRANSFORMERS = False
HAS_PYGAME = False
HAS_SPEECH_RECOGNITION = False

app = Flask(__name__)
CORS(app)


class ChatbotAPI:
    """Chatbot API مخفف مع كل المزايا."""

    def __init__(self):
        print("🤖 Initializing Lightweight Chatbot API...")

        # File paths
        self.dataset_file = 'dataset.csv'

        # Settings
        self.confidence_threshold = 0.2  # خفض threshold علشان العربية
        self.auto_translate = False

        # Data storage
        self.questions = []
        self.answers = []
        self.vectorizer = None
        self.embeddings = None
        self.is_trained = False

        # Initialize components
        self.init_translator()
        self.load_data()
        self.train_model()

        print("✅ Lightweight Chatbot API initialized successfully")

    def init_translator(self):
        """Initialize translator."""
        if HAS_TRANSLATOR:
            try:
                self.translator = Translator()
                print("🌍 Translator initialized")
            except Exception as e:
                print(f"⚠️ Translator initialization failed: {e}")
                self.translator = None

    def detect_language(self, text):
        """Detect language using regex pattern - محسنة للعربية."""
        if not text or not isinstance(text, str):
            return 'en'

        # نمط عربي شامل
        arabic_pattern = re.compile(
            r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')

        # كلمات إنجليزية شائعة
        english_words = ['hello', 'hi', 'how', 'what', 'when',
                         'where', 'why', 'thank', 'good', 'bye', 'yes', 'no', 'ok']
        text_lower = text.lower().strip()

        # إذا النص يحتوي على حروف عربية
        if arabic_pattern.search(text):
            return 'ar'
        # إذا النص يحتوي على كلمات إنجليزية شائعة
        elif any(word in text_lower for word in english_words):
            return 'en'
        else:
            # إذا مفيش مؤشر واضح، استخدم إحصاء الحروف
            arabic_chars = len(arabic_pattern.findall(text))
            total_chars = len(text)

            if arabic_chars / total_chars > 0.3:  # إذا أكثر من 30% عربية
                return 'ar'
            else:
                return 'en'

    def load_data(self):
        """Load dataset from CSV - يحافظ على البيانات الموجودة."""
        if os.path.exists(self.dataset_file):
            try:
                # حاول قراءة الملف بطرق مختلفة
                try:
                    df = pd.read_csv(self.dataset_file, encoding='utf-8')
                except:
                    df = pd.read_csv(self.dataset_file, encoding='utf-8', on_bad_lines='skip')
                
                # تحقق من وجود الأعمدة المطلوبة
                if len(df.columns) >= 2:
                    # خذ أول عمودين فقط لو في أكتر من كده
                    if 'question' in df.columns and 'answer' in df.columns:
                        questions_col = 'question'
                        answers_col = 'answer'
                    else:
                        questions_col = df.columns[0]
                        answers_col = df.columns[1]
                        print(f"⚠️ Using columns: {questions_col} and {answers_col}")
                    
                    # نظف البيانات
                    df_clean = df[[questions_col, answers_col]].copy()
                    df_clean = df_clean.dropna()
                    df_clean[questions_col] = df_clean[questions_col].astype(str)
                    df_clean[answers_col] = df_clean[answers_col].astype(str)
                    
                    # إزالة الصفوف الفارغة
                    df_clean = df_clean[(df_clean[questions_col].str.strip() != '') & 
                                      (df_clean[answers_col].str.strip() != '')]
                    
                    if len(df_clean) > 0:
                        self.questions = df_clean[questions_col].tolist()
                        self.answers = df_clean[answers_col].tolist()
                        
                        # إحصاء الأسئلة العربية
                        arabic_count = sum(
                            1 for q in self.questions if self.detect_language(q) == 'ar')
                        print(
                            f"📊 Loaded {len(self.questions)} Q&A pairs from dataset ({arabic_count} Arabic)")
                        return
                    else:
                        print("⚠️ Dataset file exists but contains no valid data")
                else:
                    print("⚠️ Dataset file doesn't have enough columns")
                    
            except Exception as e:
                print(f"⚠️ Error loading dataset: {e}")
                # حاول إصلاح الملف
                self.try_fix_dataset()
                return

        # فقط إذا الملف مش موجود أو مش ممكن نصلحه
        print("📝 Creating default dataset...")
        self.create_default_dataset()

    def try_fix_dataset(self):
        """محاولة إصلاح ملف dataset بدل حذفه."""
        try:
            print("🛠️ Attempting to fix dataset file...")
            
            # حاول قراءة الملف بطرق مختلفة
            try:
                df = pd.read_csv(self.dataset_file, encoding='utf-8', on_bad_lines='skip')
            except:
                try:
                    df = pd.read_csv(self.dataset_file, encoding='utf-8', sep=None, engine='python')
                except:
                    print("❌ Could not read dataset file")
                    self.create_default_dataset()
                    return
            
            if len(df.columns) >= 2:
                # خذ أول عمودين فقط
                df_fixed = df.iloc[:, :2].copy()
                df_fixed.columns = ['question', 'answer']
                
                # نظف البيانات
                df_fixed = df_fixed.dropna()
                df_fixed['question'] = df_fixed['question'].astype(str)
                df_fixed['answer'] = df_fixed['answer'].astype(str)
                
                # إزالة الصفوف الفارغة
                df_fixed = df_fixed[(df_fixed['question'].str.strip() != '') & 
                                  (df_fixed['answer'].str.strip() != '')]
                
                if len(df_fixed) > 0:
                    # احفظ الملف المُصلح
                    df_fixed.to_csv(self.dataset_file, index=False, encoding='utf-8')
                    print(f"✅ Fixed dataset file, keeping {len(df_fixed)} Q&A pairs")
                    
                    # حمّل البيانات المُصلحة
                    self.questions = df_fixed['question'].tolist()
                    self.answers = df_fixed['answer'].tolist()
                    return
            
            print("❌ Could not fix dataset file, creating new one")
            self.create_default_dataset()
            
        except Exception as e:
            print(f"❌ Error fixing dataset: {e}")
            self.create_default_dataset()

    def create_default_dataset(self):
        """Create default dataset فقط إذا مفيش بيانات."""
        # إذا في بيانات موجودة، ما تنشئش بيانات افتراضية
        if len(self.questions) > 0 and len(self.answers) > 0:
            print("📊 Using existing data, skipping default dataset creation")
            return
            
        data = {
            'question': [
                'Hello', 'Hi', 'How are you?', 'What is your name?', 'Goodbye', 'Thank you',
                'What can you do?', 'Help me', 'Good morning', 'Good evening',
                'مرحبا', 'أهلا', 'كيف حالك؟', 'ما اسمك؟', 'مع السلامة', 'شكرا',
                'ماذا تستطيع أن تفعل؟', 'ساعدني', 'صباح الخير', 'مساء الخير'
            ],
            'answer': [
                'Hello! How can I help you today?',
                'Hi there! Nice to meet you!',
                'I am doing well, thank you! How are you?',
                'I am an AI chatbot here to help you.',
                'Goodbye! Have a great day!',
                'You are welcome! Happy to help!',
                'I can answer questions and have conversations in English and Arabic.',
                'I am here to help! What do you need?',
                'Good morning! Hope you have a wonderful day!',
                'Good evening! How can I assist you?',
                'مرحبا! كيف يمكنني مساعدتك اليوم؟',
                'أهلا! سعيد بلقائك!',
                'أنا بخير، شكرا لك! كيف حالك؟',
                'أنا مساعد ذكي هنا لمساعدتك.',
                'مع السلامة! أتمنى لك يوما رائعا!',
                'عفوًا! سعيد بمساعدتك!',
                'يمكنني الإجابة على الأسئلة وإجراء محادثات باللغتين الإنجليزية والعربية.',
                'أنا هنا لمساعدتك! ما الذي تحتاجه؟',
                'صباح الخير! أتمنى لك يوما رائعا!',
                'مساء الخير! كيف يمكنني مساعدتك؟'
            ]
        }

        df = pd.DataFrame(data)
        df.to_csv(self.dataset_file, index=False, encoding='utf-8')
        self.questions = data['question']
        self.answers = data['answer']

        arabic_count = sum(
            1 for q in self.questions if self.detect_language(q) == 'ar')
        print(
            f"📝 Created default dataset with {len(self.questions)} Q&A pairs ({arabic_count} Arabic)")

    def train_model(self):
        """Train model using TF-IDF مع دعم محسن للعربية."""
        if not HAS_SKLEARN or not self.questions:
            print("❌ sklearn not available or no questions")
            return False

        try:
            print("🔄 Training TF-IDF model with Arabic support...")

            # استخدام إعدادات تدعم العربية
            self.vectorizer = TfidfVectorizer(
                lowercase=False,  # لا تحول للأحرف الصغيرة علشان العربية
                analyzer='char_wb',  # استخدام الأحرف
                ngram_range=(2, 4),  # نطاق أوسع للتعرف على الأنماط
                max_features=2000
            )

            self.embeddings = self.vectorizer.fit_transform(self.questions)
            self.is_trained = True

            print(f"✅ TF-IDF model training completed")
            print(
                f"   - Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
            print(
                f"   - Arabic questions: {sum(1 for q in self.questions if self.detect_language(q) == 'ar')}")
            print(
                f"   - English questions: {sum(1 for q in self.questions if self.detect_language(q) == 'en')}")

            return True
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return False

    def get_answer(self, question):
        """Get answer for a question using TF-IDF مع دعم محسن للعربية."""
        if not self.is_trained or not HAS_SKLEARN:
            if self.detect_language(question) == 'ar':
                return {
                    'answer': 'النموذج غير جاهز بعد.',
                    'confidence': 0.0,
                    'language': 'ar'
                }
            else:
                return {
                    'answer': 'Sorry, the model is not ready yet.',
                    'confidence': 0.0,
                    'language': 'en'
                }

        language = self.detect_language(question)
        print(f"💬 Processing: '{question}' (language: {language})")

        try:
            # استخدام TF-IDF بدل sentence-transformers
            question_vec = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, self.embeddings)[0]

            best_idx = np.argmax(similarities)
            best_confidence = float(similarities[best_idx])
            best_answer = self.answers[best_idx]

            print(
                f"🎯 Best match: {best_confidence:.3f} confidence -> '{best_answer}'")

            if best_confidence < self.confidence_threshold:
                if language == 'ar':
                    return {
                        'answer': 'عذراً، لم أفهم سؤالك.',
                        'confidence': best_confidence,
                        'language': 'ar'
                    }
                else:
                    return {
                        'answer': 'Sorry, I do not understand your question.',
                        'confidence': best_confidence,
                        'language': 'en'
                    }

            # الترجمة إذا needed
            answer_language = self.detect_language(best_answer)
            final_answer = best_answer

            if not self.auto_translate and language != answer_language and self.translator:
                try:
                    translated = self.translator.translate(
                        best_answer,
                        dest='ar' if language == 'ar' else 'en'
                    )
                    if translated.text:
                        final_answer = translated.text
                        print(f"🌍 Translated answer to {language}")
                except Exception as e:
                    print(f"⚠️ Translation failed: {e}")

            return {
                'answer': final_answer,
                'confidence': best_confidence,
                'language': language
            }

        except Exception as e:
            print(f"❌ Error getting answer: {e}")
            if self.detect_language(question) == 'ar':
                return {
                    'answer': 'حدث خطأ، الرجاء المحاولة مرة أخرى.',
                    'confidence': 0.0,
                    'language': 'ar'
                }
            else:
                return {
                    'answer': 'Sorry, an error occurred.',
                    'confidence': 0.0,
                    'language': 'en'
                }

    def generate_speech(self, text, language='en'):
        """Generate speech audio file."""
        if not HAS_GTTS:
            print("❌ gTTS not available")
            return None

        try:
            # تحديد لغة gTTS بناءً على اللغة المكتشفة
            tts_lang = 'ar' if self.detect_language(text) == 'ar' else language
            tts = gTTS(text=text, lang=tts_lang, slow=False)
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.mp3')
            tts.save(temp_file.name)
            print(f"🔊 Generated speech: {temp_file.name} (lang: {tts_lang})")
            return temp_file.name
        except Exception as e:
            print(f"❌ Speech generation failed: {e}")
            return None

    def process_voice_input(self, audio_file_path):
        """Voice processing simulation - استخدام Web Speech API في المتصفح."""
        print("🎤 Voice processing: Use browser Web Speech API")
        return "Voice recognition: Use browser speech API"


# Initialize chatbot
chatbot = ChatbotAPI()

# كل واجهات API تبقى كما هي بدون تغيير


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_ready': chatbot.is_trained,
        'questions_count': len(chatbot.questions),
        'arabic_questions': sum(1 for q in chatbot.questions if chatbot.detect_language(q) == 'ar'),
        'english_questions': sum(1 for q in chatbot.questions if chatbot.detect_language(q) == 'en'),
        'timestamp': datetime.datetime.now().isoformat()
    })


@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Message is required'}), 400

        result = chatbot.get_answer(message)
        response = {
            'answer': str(result['answer']),
            'confidence': float(result['confidence']),
            'language': str(result['language']),
            'timestamp': datetime.datetime.now().isoformat()
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/speech', methods=['POST'])
def speech_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        text = data.get('text', '').strip()
        language = data.get('language', 'en')
        if not text:
            return jsonify({'error': 'Text is required'}), 400

        audio_file = chatbot.generate_speech(text, language)
        if audio_file:
            try:
                response = send_file(
                    audio_file, as_attachment=True, download_name='speech.mp3')

                @response.call_on_close
                def cleanup():
                    try:
                        if os.path.exists(audio_file):
                            os.unlink(audio_file)
                    except:
                        pass
                return response
            except Exception as e:
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
                raise e
        else:
            return jsonify({'error': 'Speech generation failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice-input', methods=['POST'])
def voice_input_endpoint():
    """استخدام Web Speech API في المتصفح للتعرف على الصوت."""
    return jsonify({
        'message': 'Use browser Web Speech API for voice recognition',
        'supported': True,
        'instruction': 'Use window.SpeechRecognition in browser'
    })


@app.route('/api/dataset', methods=['GET', 'POST'])
def dataset_endpoint():
    try:
        if request.method == 'GET':
            return jsonify({
                'questions': [str(q) for q in chatbot.questions],
                'answers': [str(a) for a in chatbot.answers],
                'count': len(chatbot.questions),
                'arabic_count': sum(1 for q in chatbot.questions if chatbot.detect_language(q) == 'ar'),
                'english_count': sum(1 for q in chatbot.questions if chatbot.detect_language(q) == 'en')
            })
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400

            question = data.get('question', '').strip()
            answer = data.get('answer', '').strip()
            if not question or not answer:
                return jsonify({'error': 'Question and answer are required'}), 400

            chatbot.questions.append(question)
            chatbot.answers.append(answer)
            df = pd.DataFrame(
                {'question': chatbot.questions, 'answer': chatbot.answers})
            df.to_csv(chatbot.dataset_file, index=False, encoding='utf-8')
            chatbot.train_model()

            return jsonify({
                'success': True,
                'new_count': len(chatbot.questions),
                'language': chatbot.detect_language(question),
                'message': 'Question added successfully'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status_endpoint():
    return jsonify({
        'model_trained': bool(chatbot.is_trained),
        'dataset_size': int(len(chatbot.questions)),
        'arabic_questions': sum(1 for q in chatbot.questions if chatbot.detect_language(q) == 'ar'),
        'english_questions': sum(1 for q in chatbot.questions if chatbot.detect_language(q) == 'en'),
        'features': {
            'text_to_speech': bool(HAS_GTTS),
            'speech_recognition': False,  # استخدام المتصفح
            'translation': bool(HAS_TRANSLATOR and chatbot.translator is not None),
            'browser_voice_supported': True,
            'arabic_support': True
        }
    })


@app.route('/api/conversation', methods=['POST'])
def conversation_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        message = data.get('message', '').strip()
        get_speech = data.get('get_speech', True)
        if not message:
            return jsonify({'error': 'Message is required'}), 400

        result = chatbot.get_answer(message)
        response = {
            'answer': str(result['answer']),
            'confidence': float(result['confidence']),
            'language': str(result['language']),
            'timestamp': datetime.datetime.now().isoformat()
        }

        if get_speech and HAS_GTTS:
            audio_file = chatbot.generate_speech(
                result['answer'], result['language'])
            if audio_file:
                response['speech_url'] = f'/api/speech-file?text={result["answer"]}&lang={result["language"]}'

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/speech-file', methods=['GET'])
def speech_file_endpoint():
    try:
        text = request.args.get('text', '')
        lang = request.args.get('lang', 'en')
        if not text:
            return jsonify({'error': 'Text parameter is required'}), 400

        audio_file = chatbot.generate_speech(text, lang)
        if audio_file:
            response = send_file(
                audio_file, as_attachment=False, download_name='speech.mp3')

            @response.call_on_close
            def cleanup():
                try:
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
                except:
                    pass
            return response
        else:
            return jsonify({'error': 'Speech generation failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/browser-voice-support', methods=['GET'])
def browser_voice_support():
    return jsonify({
        'browser_voice_supported': True,
        'text_to_speech': 'Use window.speechSynthesis',
        'speech_recognition': 'Use window.SpeechRecognition',
        'languages': ['en-US', 'ar-SA', 'en-GB'],
        'arabic_support': True
    })

# Error handlers


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print("🚀 Starting Lightweight Flask API server...")
    print(f"🌐 Server will run on port: {port}")
    print("📡 All API endpoints are available!")
    app.run(debug=False, host='0.0.0.0', port=port)
