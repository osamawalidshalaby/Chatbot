<<<<<<< HEAD
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
        """Load dataset from CSV."""
        if os.path.exists(self.dataset_file):
            try:
                df = pd.read_csv(self.dataset_file, encoding='utf-8')
                if 'question' in df.columns and 'answer' in df.columns:
                    self.questions = df['question'].fillna(
                        '').astype(str).tolist()
                    self.answers = df['answer'].fillna('').astype(str).tolist()

                    # إحصاء الأسئلة العربية
                    arabic_count = sum(
                        1 for q in self.questions if self.detect_language(q) == 'ar')
                    print(
                        f"📊 Loaded {len(self.questions)} Q&A pairs ({arabic_count} Arabic)")
                    return
            except Exception as e:
                print(f"⚠️ Error loading dataset: {e}")

        # إنشاء بيانات افتراضية إذا الملف مش موجود
        self.create_default_dataset()

    def create_default_dataset(self):
        """Create default dataset."""
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
=======

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import datetime
import re
import pickle
import tempfile
import time
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
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
    import pygame
    HAS_PYGAME = True
    pygame.mixer.init()
except ImportError:
    HAS_PYGAME = False

try:
    from googletrans import Translator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False

app = Flask(__name__)
CORS(app)  # Enable CORS for React app


class ChatbotAPI:
    """Chatbot API for React integration."""

    def __init__(self):
        print("🤖 Initializing Chatbot API...")

        # File paths
        self.dataset_file = 'dataset.csv'
        self.embeddings_file = 'embeddings.pkl'

        # Settings
        self.confidence_threshold = 0.3
        self.auto_translate = False

        # Data storage
        self.questions = []
        self.answers = []
        self.embeddings = None
        self.is_trained = False

        # Speech recognition
        self.recognizer = None
        self.microphone = None

        # Initialize components
        self.init_sentence_model()
        self.init_speech_recognition()
        self.init_translator()

        # Language detection
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF]+')

        # Load data and train
        self.load_data()
        self.train_model()

        print("✅ Chatbot API initialized successfully")

    def init_sentence_model(self):
        """Initialize the sentence transformer model."""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                print("🧠 Loading sentence model...")
                self.sentence_model = SentenceTransformer(
                    'paraphrase-multilingual-MiniLM-L12-v2')
            except Exception as e:
                print(f"❌ Failed to load sentence model: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None

    def init_speech_recognition(self):
        """Initialize speech recognition."""
        if HAS_SPEECH_RECOGNITION:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(
                        source, duration=1)
                print("🎤 Speech recognition initialized")
            except Exception as e:
                print(f"⚠️ Speech recognition initialization failed: {e}")
                self.recognizer = None
                self.microphone = None

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
        """Detect language using regex pattern."""
        if not text:
            return 'en'
        return 'ar' if self.arabic_pattern.search(text) else 'en'

    def load_data(self):
        """Load dataset from CSV or create default one."""
        if os.path.exists(self.dataset_file):
            try:
                df = pd.read_csv(self.dataset_file, encoding='utf-8')
                if 'question' in df.columns and 'answer' in df.columns:
                    self.questions = df['question'].fillna(
                        '').astype(str).tolist()
                    self.answers = df['answer'].fillna('').astype(str).tolist()
                    print(
                        f"📊 Loaded {len(self.questions)} Q&A pairs from dataset")
                    return
            except Exception as e:
                print(f"⚠️ Error loading dataset: {e}")

        self.create_default_dataset()

    def create_default_dataset(self):
        """Create a default dataset."""
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
        print("📝 Created default dataset with 20 Q&A pairs")

    def train_model(self):
        """Train model or load existing embeddings."""
        if not self.sentence_model or not HAS_SKLEARN:
            print("❌ Sentence model or sklearn not available")
            return False

        if not self.questions:
            print("❌ No questions available for training")
            return False

        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    saved_data = pickle.load(f)

                if (saved_data.get('questions') == self.questions and
                        saved_data.get('answers') == self.answers):
                    self.embeddings = saved_data['embeddings']
                    self.is_trained = True
                    print("📚 Loaded existing embeddings")
                    return True
            except Exception as e:
                print(f"⚠️ Error loading embeddings: {e}")

        try:
            print("🔄 Training model with new embeddings...")
            self.embeddings = self.sentence_model.encode(self.questions)
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump({
                    'questions': self.questions,
                    'answers': self.answers,
                    'embeddings': self.embeddings
                }, f)
            self.is_trained = True
            print("✅ Model training completed")
            return True
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return False

    def get_answer(self, question):
        """Get answer for a question."""
        if not self.is_trained or not HAS_SKLEARN:
            return {
                'answer': 'Sorry, the model is not ready yet.',
                'confidence': 0.0,
                'language': 'en'
            }

        language = self.detect_language(question)
        print(f"💬 Processing question: '{question}' (language: {language})")

        try:
            question_embedding = self.sentence_model.encode([question])
            similarities = cosine_similarity(
                question_embedding, self.embeddings)[0]

            best_idx = np.argmax(similarities)
            # Convert to Python float
            best_confidence = float(similarities[best_idx])
            best_answer = self.answers[best_idx]

            print(f"🎯 Best match: {best_confidence:.3f} confidence")

            if best_confidence < self.confidence_threshold:
                return {
                    'answer': 'Sorry, I do not understand your question.',
                    'confidence': best_confidence,
                    'language': 'en'
                }

            answer_language = self.detect_language(best_answer)
            final_answer = best_answer

            if not self.auto_translate and language != answer_language and self.translator:
                try:
                    translated = self.translator.translate(best_answer,
                                                           dest='ar' if language == 'ar' else 'en')
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
            tts = gTTS(text=text, lang=language, slow=False)
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.mp3')
            tts.save(temp_file.name)
            print(f"🔊 Generated speech: {temp_file.name}")
            return temp_file.name
        except Exception as e:
            print(f"❌ Speech generation failed: {e}")
            return None

    # Add this improved voice processing method to your ChatbotAPI class in app.py


    def process_voice_input(self, audio_file_path):
        """Process voice input from audio file with better error handling."""
        if not self.recognizer or not HAS_SPEECH_RECOGNITION:
            print("❌ Speech recognition not available")
            return None
    
        try:
            print(f"🎤 Processing audio file: {audio_file_path}")
    
            # Check file size and duration
            file_size = os.path.getsize(audio_file_path)
            if file_size < 1000:  # Less than 1KB
                print("❌ Audio file too small")
                return None
    
            with sr.AudioFile(audio_file_path) as source:
                # Adjust for ambient noise with longer duration
                print("🔊 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("🎙️ Recording audio...")
    
                # Read the entire audio file
                audio = self.recognizer.record(source)
    
                # Check audio duration
                duration = len(audio.frame_data) / \
                    (audio.sample_rate * audio.sample_width)
                print(f"⏱️ Audio duration: {duration:.2f} seconds")
    
                if duration < 0.5:
                    print("❌ Audio too short")
                    return None
                if duration > 30:
                    print("❌ Audio too long")
                    return None
    
            print("🔄 Recognizing speech...")
    
            # Try multiple recognition engines and languages
            recognition_methods = [
                # Try Google with Arabic
                {'method': 'google', 'language': 'ar-AR', 'name': 'Google Arabic'},
                # Try Google with English
                {'method': 'google', 'language': 'en-US', 'name': 'Google English'},
                # Try Google with auto-detect
                {'method': 'google', 'language': None, 'name': 'Google Auto'},
            ]
    
            for method in recognition_methods:
                try:
                    if method['method'] == 'google':
                        text = self.recognizer.recognize_google(
                            audio,
                            language=method['language']
                        )
                        print(f"✅ {method['name']} recognition: '{text}'")
                        return text
    
                except sr.UnknownValueError:
                    print(f"❌ {method['name']} could not understand audio")
                    continue
                except sr.RequestError as e:
                    print(f"❌ {method['name']} error: {e}")
                    continue
                except Exception as e:
                    print(f"❌ {method['name']} unexpected error: {e}")
                    continue
                
            print("❌ All recognition methods failed")
            return None
    
        except Exception as e:
            print(f"❌ Voice processing error: {str(e)}")
            return None
    
    # Initialize chatbot
chatbot = ChatbotAPI()
    
# API Routes


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_ready': chatbot.is_trained,
        'questions_count': len(chatbot.questions),
        'timestamp': datetime.datetime.now().isoformat()
    })


@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Chat endpoint for text input."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        message = data.get('message', '').strip()

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        result = chatbot.get_answer(message)

        # Ensure all values are JSON serializable
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
    """Generate speech from text."""
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
            # Clean up the file after sending
            try:
                response = send_file(
                    audio_file, as_attachment=True, download_name='speech.mp3')

                # Schedule file cleanup after response is sent
                @response.call_on_close
                def cleanup():
                    try:
                        if os.path.exists(audio_file):
                            os.unlink(audio_file)
                            print("🧹 Cleaned up speech file")
                    except:
                        pass

                return response
            except Exception as e:
                # Clean up if there's an error
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
                raise e
        else:
            return jsonify({'error': 'Speech generation failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice-input', methods=['POST'])
def voice_input_endpoint():
    """Process voice input from audio file."""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Audio file is required', 'success': False}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400

        print(f"📁 Received audio file: {audio_file.filename}")

        # Validate file type
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac']
        file_ext = os.path.splitext(audio_file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'File type {file_ext} not supported. Use: {", ".join(allowed_extensions)}',
                'success': False
            }), 400

        # Save uploaded file temporarily
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        audio_file.save(temp_audio.name)
        print(f"💾 Saved temporary file: {temp_audio.name}")

        # Process voice input
        text = chatbot.process_voice_input(temp_audio.name)

        # Cleanup
        try:
            os.unlink(temp_audio.name)
            print("🧹 Cleaned up temporary file")
        except Exception as e:
            print(f"⚠️ Could not delete temp file: {e}")

        if text:
            return jsonify({
                'text': str(text),
                'language': str(chatbot.detect_language(text)),
                'success': True
            })
        else:
            return jsonify({
                'error': 'Could not recognize speech from audio',
                'success': False
            }), 400

    except Exception as e:
        print(f"❌ Voice endpoint error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/dataset', methods=['GET', 'POST'])
def dataset_endpoint():
    """Manage dataset."""
    try:
        if request.method == 'GET':
            return jsonify({
                'questions': [str(q) for q in chatbot.questions],
                'answers': [str(a) for a in chatbot.answers],
                'count': len(chatbot.questions)
            })

        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400

            question = data.get('question', '').strip()
            answer = data.get('answer', '').strip()

            if not question or not answer:
                return jsonify({'error': 'Question and answer are required'}), 400

            # Add to dataset
            chatbot.questions.append(question)
            chatbot.answers.append(answer)

            # Update CSV
            df = pd.DataFrame({
                'question': chatbot.questions,
                'answer': chatbot.answers
            })
            df.to_csv(chatbot.dataset_file, index=False, encoding='utf-8')

            # Retrain model
            chatbot.train_model()

            return jsonify({
                'success': True,
                'new_count': len(chatbot.questions),
                'message': 'Question added successfully'
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status_endpoint():
    """Get chatbot status."""
    return jsonify({
        'model_trained': bool(chatbot.is_trained),
        'dataset_size': int(len(chatbot.questions)),
        'features': {
            'text_to_speech': bool(HAS_GTTS),
            'speech_recognition': bool(HAS_SPEECH_RECOGNITION and chatbot.recognizer is not None),
            'translation': bool(HAS_TRANSLATOR and chatbot.translator is not None),
            'browser_voice_supported': True  # Always true for browser voice
        }
    })


@app.route('/api/conversation', methods=['POST'])
def conversation_endpoint():
    """Handle complete conversation with text and speech."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        message = data.get('message', '').strip()
        get_speech = data.get('get_speech', True)

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Get text response
        result = chatbot.get_answer(message)

        response = {
            'answer': str(result['answer']),
            'confidence': float(result['confidence']),
            'language': str(result['language']),
            'timestamp': datetime.datetime.now().isoformat()
        }

        # Generate speech if requested
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
    """Get speech file directly."""
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
                        print("🧹 Cleaned up speech file")
                except:
                    pass

            return response
        else:
            return jsonify({'error': 'Speech generation failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/audio-test', methods=['POST'])
def audio_test_endpoint():
    """Test audio processing with sample data."""
    try:
        # Test with a simple English phrase
        test_text = "Hello this is a test"

        # Generate test audio
        if HAS_GTTS:
            audio_file = chatbot.generate_speech(test_text, 'en')
            if audio_file:
                # Try to process the generated audio
                processed_text = chatbot.process_voice_input(audio_file)

                # Cleanup
                try:
                    os.unlink(audio_file)
                except:
                    pass

                return jsonify({
                    'original_text': test_text,
                    'processed_text': processed_text,
                    'match': test_text.lower() == processed_text.lower() if processed_text else False,
                    'success': True
                })

        return jsonify({
            'error': 'Audio test failed - check dependencies',
            'success': False
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/browser-voice-support', methods=['GET'])
def browser_voice_support():
    """Check browser voice recognition support."""
    return jsonify({
        'browser_voice_supported': True,
        'message': 'Use Web Speech API in browser for voice recognition',
        'instructions': {
            'start_listening': 'Use navigator.mediaDevices.getUserMedia() for audio recording',
            'stop_listening': 'Stop the media recorder and send audio to /api/voice-input',
            'browser_speech_api': 'Use window.SpeechRecognition or window.webkitSpeechRecognition'
        }
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


# في نهاية الملف، استبدل if __name__ == '__main__' بالكود ده:

if __name__ == '__main__':
    # احصل على البورت من environment variable أو استخدم 8000
    port = int(os.environ.get('PORT', 8000))

    print("🚀 Starting Flask API server...")
    print(f"🌐 Server will run on port: {port}")
    print("📡 API endpoints:")
    print("   GET  /api/health              - Health check")
    print("   POST /api/chat                - Send text message")
    print("   POST /api/speech              - Generate speech")
    print("   POST /api/voice-input         - Process voice input")
    print("   GET  /api/dataset             - Get dataset")
    print("   POST /api/dataset             - Add Q&A pair")
    print("   GET  /api/status              - Get status")
    print("   POST /api/conversation        - Complete conversation")
    print("   GET  /api/speech-file         - Get speech file")
    print("   POST /api/audio-test          - Test audio features")
    print("   GET  /api/browser-voice-support - Browser voice support info")
    print("🎤 Voice Features:")
    print("   - Text-to-Speech: Available" if HAS_GTTS else "   - Text-to-Speech: Not available")
    print("   - Speech Recognition: Available" if HAS_SPEECH_RECOGNITION else "   - Speech Recognition: Not available")
    print("   - Browser Voice: Always available via Web Speech API")

    # استخدم debug=False للسيرفر
    app.run(debug=False, host='0.0.0.0', port=port)


>>>>>>> 2e5776350d56de9f2773f72e3a61d5910f82b97a
