import re
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os
import requests

# --- Handle Missing Dependencies ---
MISSING_DEPENDENCIES = []

try:
    import mysql.connector
except ImportError:
    MISSING_DEPENDENCIES.append("mysql-connector-python")
    print("⚠️ MySQL connector not found. Database features will be disabled.")

if MISSING_DEPENDENCIES:
    print(f"\n❌ Missing dependencies: {', '.join(MISSING_DEPENDENCIES)}")
    print("To install, run: pip install " + " ".join(MISSING_DEPENDENCIES))

# --- Create Directories ---
TRUE_FOLDER = "True"
FAKE_FOLDER = "Fake"
MODELS_DIR = "models"

if not os.path.exists(TRUE_FOLDER):
    os.makedirs(TRUE_FOLDER)
if not os.path.exists(FAKE_FOLDER):
    os.makedirs(FAKE_FOLDER)
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# --- DB Connection (Optional) ---
cursor = None
db = None

def setup_database():
    global db, cursor
    try:
        db = mysql.connector.connect(host="localhost", user="root", password="qwerty00", database="fake_news_project")
        cursor = db.cursor()
        print("✅ Database connection established.")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("⚠️ Will proceed without database features. Using folder and CSV data only.")
        return False

# --- API Key ---
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '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')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Helper Functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# --- Model Training ---
def train_models(include_groq_feedback=False, groq_text=None, groq_label=None, extra_weight=1):
    additional_real = []
    additional_fake = []
    
    try:
        with open(os.path.join(MODELS_DIR, 'additional_examples.pkl'), 'rb') as f:
            additional_examples = pickle.load(f)
            additional_real = additional_examples.get('real', [])
            additional_fake = additional_examples.get('fake', [])
            print(f"✅ Loaded {len(additional_real)} additional real news and {len(additional_fake)} additional fake news examples")
    except Exception as e:
        print(f"ℹ️ No additional examples found or error loading: {e}")
    
    real_news = []
    fake_news = []

    if os.path.exists(TRUE_FOLDER):
        print(f"ℹ️ Loading real news from folder: {TRUE_FOLDER}")
        for filename in os.listdir(TRUE_FOLDER):
            file_path = os.path.join(TRUE_FOLDER, filename)
            try:
                if filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            real_news.append(text)
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    for col in ['title', 'text']:
                        if col in df.columns:
                            texts = df[col].dropna().tolist()
                            real_news.extend([str(t) for t in texts if str(t).strip()])
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
        print(f"✅ Read {len(real_news)} real news articles from {TRUE_FOLDER}")
    else:
        print(f"⚠️ True folder not found at {TRUE_FOLDER}")

    if os.path.exists(FAKE_FOLDER):
        print(f"ℹ️ Loading fake news from folder: {FAKE_FOLDER}")
        for filename in os.listdir(FAKE_FOLDER):
            file_path = os.path.join(FAKE_FOLDER, filename)
            try:
                if filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            fake_news.append(text)
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    for col in ['title', 'text']:
                        if col in df.columns:
                            texts = df[col].dropna().tolist()
                            fake_news.extend([str(t) for t in texts if str(t).strip()])
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
        print(f"✅ Read {len(fake_news)} fake news articles from {FAKE_FOLDER}")
    else:
        print(f"⚠️ Fake folder not found at {FAKE_FOLDER}")

    if cursor:
        try:
            cursor.execute("SELECT title, description, content FROM news_articles")
            records = cursor.fetchall()
            db_real_news = [' '.join(filter(None, r)) for r in records if any(r)]
            real_news.extend(db_real_news)
            print(f"✅ Loaded {len(db_real_news)} real news articles from database")
        except Exception as e:
            print(f"❌ Error loading from database: {e}")

    if not real_news:
        print("ℹ️ No real news from True folder. Trying True.csv...")
        try:
            true_df = pd.read_csv("True.csv")
            if not true_df.empty:
                real_news = true_df['title'].dropna().tolist()
                print(f"✅ Loaded {len(real_news)} real news articles from True.csv")
        except Exception as e:
            print(f"❌ Error loading True.csv: {e}")
            print("⚠️ No real news data found. Creating dummy data for demo purposes.")
            real_news = [
                "NASA Confirms Evidence of Water on Mars",
                "Scientists Develop New Cancer Treatment",
                "Federal Reserve Announces Interest Rate Decision",
                "New Study Links Exercise to Longevity",
                "Stock Market Closes at Record High"
            ]
            print(f"✅ Loaded {len(real_news)} real news articles from dummy data")

    if not fake_news:
        print("ℹ️ No fake news from Fake folder. Trying Fake.csv...")
        try:
            fake_df = pd.read_csv("Fake.csv")
            fake_news = fake_df['title'].dropna().tolist()
            print(f"✅ Loaded {len(fake_news)} fake news articles from Fake.csv")
        except Exception as e:
            print(f"❌ Error loading Fake.csv: {e}")
            print("⚠️ No fake news data found. Creating dummy data for demo purposes.")
            fake_news = [
                "Aliens Make Contact With Government Officials",
                "Miracle Cure Discovered That Big Pharma Is Hiding",
                "Scientists Confirm the Earth is Actually Flat",
                "5G Networks Secretly Controlling Minds",
                "Celebrity Secretly Replaced by Clone"
            ]
            print(f"✅ Loaded {len(fake_news)} fake news articles from dummy data")

    real_news.extend(additional_real)
    fake_news.extend(additional_fake)
    
    if include_groq_feedback and groq_text and groq_label is not None:
        if groq_label == 1:
            additional_real.append(groq_text)
            for _ in range(extra_weight):
                real_news.append(groq_text)
        else:
            additional_fake.append(groq_text)
            for _ in range(extra_weight):
                fake_news.append(groq_text)
        print(f"✅ Added Groq feedback as {'real' if groq_label == 1 else 'fake'} news with weight {extra_weight}")
        
        try:
            with open(os.path.join(MODELS_DIR, 'additional_examples.pkl'), 'wb') as f:
                pickle.dump({'real': additional_real, 'fake': additional_fake}, f)
            print(f"✅ Saved {len(additional_real)} real and {len(additional_fake)} fake additional examples to {MODELS_DIR}/additional_examples.pkl")
        except Exception as e:
            print(f"❌ Error saving additional examples: {e}")

    min_len = min(len(real_news), len(fake_news))
    real_news = real_news[:min_len]
    fake_news = fake_news[:min_len]
    
    print(f"ℹ️ Training with {len(real_news)} real and {len(fake_news)} fake news articles")

    texts = real_news + fake_news
    labels = [1]*len(real_news) + [0]*len(fake_news)
    texts = [clean_text(t) for t in texts]

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=3, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

    print(f"ℹ️ Training set: {X_train.shape[0]} articles ({len([l for l in y_train if l == 1])} real, {len([l for l in y_train if l == 0])} fake)")
    print(f"ℹ️ Test set: {X_test.shape[0]} articles ({len([l for l in y_test if l == 1])} real, {len([l for l in y_test if l == 0])} fake)")

    print("\n🔄 Training Random Forest Classifier model...")
    start_time = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    rf_training_time = time.time() - start_time
    
    rf_preds = rf_model.predict(X_test)
    print(f"⏱️ Random Forest Classifier training completed in {rf_training_time:.2f} seconds")
    print("📊 Random Forest Classifier Results:")
    print(classification_report(y_test, rf_preds))
    
    if hasattr(rf_model, 'feature_importances_'):
        rf_feature_names = vectorizer.get_feature_names_out()
        rf_feature_importance = sorted(zip(rf_feature_names, rf_model.feature_importances_), 
                                      key=lambda x: x[1], reverse=True)
        print("\n🔍 Top 10 Most Important Features (Random Forest Classifier):")
        for feature, importance in rf_feature_importance[:10]:
            print(f"{feature}: {importance:.4f}")

    print("\n🚀 Training Logistic Regression Classifier model...")
    start_time = time.time()
    lr_model = LogisticRegression(
        C=1.0,
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs'
    )
    lr_model.fit(X_train, y_train)
    lr_training_time = time.time() - start_time
    
    lr_preds = lr_model.predict(X_test)
    print(f"⏱️ Logistic Regression Classifier training completed in {lr_training_time:.2f} seconds")
    print("📊 Logistic Regression Classifier Results:")
    print(classification_report(y_test, lr_preds))

    if hasattr(lr_model, 'coef_'):
        lr_feature_names = vectorizer.get_feature_names_out()
        lr_feature_importance = sorted(zip(lr_feature_names, abs(lr_model.coef_[0])), 
                                      key=lambda x: x[1], reverse=True)
        print("\n🔍 Top 10 Most Important Features (Logistic Regression Classifier):")
        for feature, importance in lr_feature_importance[:10]:
            print(f"{feature}: {importance:.4f}")
    
    timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S") if include_groq_feedback else ""
    try:
        with open(os.path.join(MODELS_DIR, f'vectorizer{timestamp}.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"✅ Saved vectorizer to {MODELS_DIR}/vectorizer{timestamp}.pkl")
        with open(os.path.join(MODELS_DIR, f'rf_model{timestamp}.pkl'), 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"✅ Saved Random Forest model to {MODELS_DIR}/rf_model{timestamp}.pkl")
        with open(os.path.join(MODELS_DIR, f'lr_model{timestamp}.pkl'), 'wb') as f:
            pickle.dump(lr_model, f)
        print(f"✅ Saved Logistic Regression model to {MODELS_DIR}/lr_model{timestamp}.pkl")
    except Exception as e:
        print(f"❌ Error saving models: {e}")

    return rf_model, lr_model, vectorizer

# --- Load Models ---
def load_models():
    try:
        with open(os.path.join(MODELS_DIR, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"✅ Loaded vectorizer from {MODELS_DIR}/vectorizer.pkl")
        with open(os.path.join(MODELS_DIR, 'rf_model.pkl'), 'rb') as f:
            rf_model = pickle.load(f)
        print(f"✅ Loaded Random Forest model from {MODELS_DIR}/rf_model.pkl")
        with open(os.path.join(MODELS_DIR, 'lr_model.pkl'), 'rb') as f:
            lr_model = pickle.load(f)
        print(f"✅ Loaded Logistic Regression model from {MODELS_DIR}/lr_model.pkl")
        return rf_model, lr_model, vectorizer
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("🔄 Training new models instead...")
        return train_models()

# --- Prediction Functions ---
def get_model_predictions(rf_model, lr_model, vectorizer, news_text):
    news_text = clean_text(news_text)
    vectorized = vectorizer.transform([news_text])
    
    rf_prediction = rf_model.predict(vectorized)[0]
    rf_proba = rf_model.predict_proba(vectorized)[0]
    rf_confidence = max(rf_proba)
    rf_result = "Real News" if rf_prediction == 1 else "Fake News"
    
    lr_prediction = lr_model.predict(vectorized)[0]
    lr_proba = lr_model.predict_proba(vectorized)[0]
    lr_confidence = max(lr_proba)
    lr_result = "Real News" if lr_prediction == 1 else "Fake News"
    
    return rf_result, rf_confidence, lr_result, lr_confidence

# --- AI Functions ---
def call_groq_api(messages, temperature=0.2, max_tokens=512):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1,
        "stream": False
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return "⚠️ API response is empty or invalid."
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Status code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        return None

def analyze_with_ai(news_text, is_followup=False, user_question=None):
    try:
        print("🔄 Making API call to Groq...")
        system_content = "You are a helpful assistant that evaluates whether news is real or fake."
        if is_followup:
            system_content += " You have expertise in journalistic standards, fact-checking, and media literacy."
            user_content = f"Regarding this news: '{news_text}'\n\nUser question: {user_question}"
        else:
            user_content = f"Is the following news real or fake? Please explain your reasoning: '{news_text}'"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        response = call_groq_api(
            messages=messages,
            temperature=0.2 if not is_followup else 0.3,
            max_tokens=512 if not is_followup else 1024
        )
        
        if response:
            return response
        else:
            print("⚠️ Failed to get response from Groq API")
            return perform_rule_based_analysis(news_text)

    except Exception as e:
        print(f"❌ AI Service Error: {e}")
        if not is_followup:
            return perform_rule_based_analysis(news_text)
        else:
            return f"Unable to provide detailed analysis due to AI service unavailability."

def perform_rule_based_analysis(news_text):
    news_lower = news_text.lower()
    sensationalist_words = ["shocking", "incredible", "unbelievable", "mind-blowing", 
                          "you won't believe", "secret", "conspiracy", "miracle", 
                          "amazing", "stunning", "jaw-dropping"]
    sensationalist_count = sum(1 for word in sensationalist_words if word in news_lower)
    excessive_punctuation = len(re.findall(r'[!?]{2,}', news_text)) > 0
    all_caps_words = len(re.findall(r'\b[A-Z]{3,}\b', news_text))
    has_clickbait = "click here" in news_lower or "you won't believe" in news_lower
    has_urgency = "act now" in news_lower or "limited time" in news_lower
    
    total_red_flags = sensationalist_count + excessive_punctuation + all_caps_words + has_clickbait + has_urgency
    
    analysis = [
        "Based on text analysis (without AI):",
        f"- Sensationalist language: {sensationalist_count} instances",
        f"- Excessive punctuation: {'Yes' if excessive_punctuation else 'No'}",
        f"- ALL CAPS words: {all_caps_words} instances",
        f"- Clickbait phrases: {'Present' if has_clickbait else 'None detected'}",
        f"- Urgency tactics: {'Present' if has_urgency else 'None detected'}"
    ]
    
    verdict = "Likely FAKE NEWS" if total_red_flags > 2 else "Possibly REAL NEWS"
    analysis.append(f"\nVerdict: {verdict}")
    
    return "\n".join(analysis)

def check_ai_agreement(ai_response, rf_prediction, lr_prediction):
    response_lower = ai_response.lower()
    
    if "real news" in response_lower or "news is real" in response_lower:
        ai_verdict = "Real News"
    elif "fake news" in response_lower or "news is fake" in response_lower:
        ai_verdict = "Fake News"
    else:
        real_indicators = ["credible", "legitimate", "trustworthy", "authentic", "factual"]
        fake_indicators = ["false", "misleading", "misinformation", "fabricated", "unreliable"]
        
        real_count = sum(1 for word in real_indicators if word in response_lower)
        fake_count = sum(1 for word in fake_indicators if word in response_lower)
        
        if real_count > fake_count:
            ai_verdict = "Real News"
        elif fake_count > real_count:
            ai_verdict = "Fake News"
        else:
            return False, "Uncertain"
    
    if ai_verdict == rf_prediction or ai_verdict == lr_prediction:
        print(f"✅ AI ({ai_verdict}) agrees with model prediction.")
        return True, ai_verdict
    else:
        print(f"❌ AI ({ai_verdict}) disagrees with model predictions ({rf_prediction}, {lr_prediction}).")
        return False, ai_verdict

def process_news(news_text, rf_model, lr_model, vectorizer):
    print(f"\n📰 Analyzing: {news_text}")
    start_time = time.time()
    rf_pred, rf_conf, lr_pred, lr_conf = get_model_predictions(
        rf_model, lr_model, vectorizer, news_text
    )
    prediction_time = time.time() - start_time
    
    print(f"🔄 Random Forest Classifier: {rf_pred} (Confidence: {rf_conf*100:.2f}%)")
    print(f"🚀 Logistic Regression: {lr_pred} (Confidence: {lr_conf*100:.2f}%)")
    print(f"⏱️ Prediction completed in {prediction_time:.4f} seconds")
    
    if lr_pred == "Fake News" and rf_pred == "Real News":
        print("\n⚠️ Model disagreement detected: LR says Fake News but RF says Real News")
        print("Automatically cross-checking with AI...")
        ai_check = analyze_with_ai(news_text)
        print(f"🔄 AI Analysis:\n{ai_check}")
        
        agreement, ai_verdict = check_ai_agreement(ai_check, rf_pred, lr_pred)
        
        if ai_verdict == "Fake News":
            print("\n✅ AI confirms Logistic Regression classification: This is likely Fake News")
            user_input = input("\n🧠 Would you like to retrain models with this example as Fake News? (y/n): ").lower()
            
            if user_input == 'y':
                rf_model, lr_model, vectorizer = train_models(
                    include_groq_feedback=True, 
                    groq_text=news_text, 
                    groq_label=0,
                    extra_weight=10
                )
                
                rf_pred, rf_conf, lr_pred, lr_conf = get_model_predictions(
                    rf_model, lr_model, vectorizer, news_text
                )
                print("\n📊 Retrained Model Predictions:")
                print(f"🔄 Random Forest Classifier: {rf_pred} (Confidence: {rf_conf*100:.2f}%)")
                print(f"🚀 Logistic Regression: {lr_pred} (Confidence: {lr_conf*100:.2f}%)")
                
                if rf_pred == "Real News" and input("\n⚠️ RF model still predicts Real News. Apply stronger training? (y/n): ").lower() == 'y':
                    rf_model, lr_model, vectorizer = train_models(
                        include_groq_feedback=True, 
                        groq_text=news_text, 
                        groq_label=0,
                        extra_weight=50
                    )
                    
                    rf_pred, rf_conf, lr_pred, lr_conf = get_model_predictions(
                        rf_model, lr_model, vectorizer, news_text
                    )
                    print("\n📊 Final Retrained Model Predictions:")
                    print(f"🔄 Random Forest Classifier: {rf_pred} (Confidence: {rf_conf*100:.2f}%)")
                    print(f"🚀 Logistic Regression: {lr_pred} (Confidence: {lr_conf*100:.2f}%)")
        else:
            print("\n⚠️ AI disagrees with Logistic Regression and supports RF: This might be Real News")
            print("No model retraining needed in this case.")
        
        ask_followup_question(news_text)
        return
    
    user_input = input("\n❓ Would you like to cross-check with AI? (y/n): ").lower()
    if user_input == 'y':
        print("\n🧠 Cross-checking with AI...")
        ai_check = analyze_with_ai(news_text)
        print(f"🔄 AI Analysis:\n{ai_check}")
        
        agreement, ai_verdict = check_ai_agreement(ai_check, rf_pred, lr_pred)
        
        if not agreement:
            user_input = input("\n⚠️ AI disagrees with model predictions.\nWould you like to retrain with AI feedback? (y/n): ").lower()
            if user_input == 'y':
                groq_label = 1 if ai_verdict == "Real News" else 0
                print("\n🧠 Retraining models with AI feedback...")
                rf_model, lr_model, vectorizer = train_models(
                    include_groq_feedback=True, 
                    groq_text=news_text, 
                    groq_label=groq_label
                )
                
                rf_pred, rf_conf, lr_pred, lr_conf = get_model_predictions(
                    rf_model, lr_model, vectorizer, news_text
                )
                print("\n📊 Retrained Model Predictions:")
                print(f"🔄 Random Forest Classifier: {rf_pred} (Confidence: {rf_conf*100:.2f}%)")
                print(f"🚀 Logistic Regression: {lr_pred} (Confidence: {lr_conf*100:.2f}%)")
    
    ask_followup_question(news_text)
    print("\n📰 Ready for next analysis.")

def ask_followup_question(news_text):
    user_input = input("\n❓ Would you like to ask a question about this news? (y/n): ").lower()
    if user_input == 'y':
        question = input("Enter your question: ")
        print("\n🔍 Getting answer from AI...")
        answer = analyze_with_ai(news_text, is_followup=True, user_question=question)
        print(f"🧠 AI Response:\n{answer}")

def test_groq_connection():
    print("\n🔍 Testing Groq API connection...")
    test_message = "Hello, this is a test message to check if the Groq API is working."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": test_message}
    ]
    
    response = call_groq_api(messages, temperature=0.1, max_tokens=50)
    
    if response:
        print("✅ Groq API connection successful!")
        print(f"🔄 Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        return True
    else:
        print("❌ Failed to connect to Groq API")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fake News Detection System")
    parser.add_argument("--train", action="store_true", help="Train new models")
    parser.add_argument("--force-retrain", action="store_true", help="Force training new models with all examples")
    parser.add_argument("--news", type=str, help="News text to analyze")
    parser.add_argument("--no-db", action="store_true", help="Skip database connection attempt")
    parser.add_argument("--test-groq", action="store_true", help="Test Groq API connection")
    args = parser.parse_args()
    
    print("\n🔍 Fake News Detection System")
    print("============================")
    
    if not args.no_db:
        setup_database()
    
    if args.test_groq:
        test_groq_connection()
    
    if args.train or args.force_retrain:
        print("\n🧠 Training new models from scratch...")
        rf_model, lr_model, vectorizer = train_models()
    else:
        print("\n🔄 Loading existing models...")
        rf_model, lr_model, vectorizer = load_models()
    
    if args.news:
        process_news(args.news, rf_model, lr_model, vectorizer)
    else:
        current_models = (rf_model, lr_model, vectorizer)
        while True:
            news_text = input("\n📰 Enter a news headline for prediction (or 'exit' to quit): ")
            if news_text.lower() == 'exit':
                break
            if not news_text.strip():
                print("⚠️ Please enter valid text.")
                continue
                
            process_news(news_text, *current_models)
            
            reload = input("\nReload models to apply all training data? (y/n): ").lower() == 'y'
            if reload:
                print("\n🔄 Reloading models with all training data...")
                current_models = train_models()

if __name__ == "__main__":
    groq_available = test_groq_connection()
    if not groq_available:
        print("\n⚠️ Groq API is not available or key is invalid.")
        print("The system will use rule-based analysis for AI features.")
    
    main()
