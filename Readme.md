TruthShield 🛡️

TruthShield is a powerful fake news detection system that blends machine learning, AI-driven analysis, and a Chrome browser extension to help users combat misinformation. Powered by Random Forest and Logistic Regression models, integrated with the Groq API, and supported by a Flask backend, TruthShield fetches real-time news from multiple APIs and stores them in a MySQL database. Whether you're a researcher, developer, or casual user, TruthShield empowers you to verify news authenticity with ease.

NOTE: You can download the dataset from this Google Drive link:

https://drive.google.com/drive/folders/1O4qtw9i0qCoxSd7bkgT2NeimIEv1r9qv?usp=sharing

🚀 Features

Advanced ML Models: Classifies news as real or fake using Random Forest and Logistic Regression with TF-IDF features.
AI-Powered Insights: Leverages the Groq API for in-depth analysis and follow-up question answering.
Chrome Extension: Intuitive UI with context menu analysis, result history, and customizable settings.
Real-Time News Collection: Fetches articles from NewsAPI, GNews, MediaStack, NewsData.io, Google RSS, CurrentsAPI, and NYTimes.
MySQL Integration: Stores news articles for training and validation.
Dynamic Retraining: Updates models with AI feedback or user-verified examples.
Flask API: Provides endpoints for programmatic analysis and querying.
Fallback Analysis: Rule-based text analysis when AI services are unavailable.


📦 Installation
Prerequisites

🐍 Python 3.8 or higher
🗄️ MySQL (optional, for database features)
🌐 Google Chrome (for the browser extension)
🔑 API keys for Groq and news APIs (see API Keys)(get the api key fro official Groq AI)

Backend Setup

Clone the Repository:
git clone https://github.com/yourusername/truthshield.git
cd truthshield


Install Dependencies:Create a virtual environment and install packages:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install flask requests mysql-connector-python pandas numpy scikit-learn pickle5 feedparser


Set Up MySQL Database (Optional):

Install MySQL and create the database:CREATE DATABASE IF NOT EXISTS fake_news_project;
USE fake_news_project;

CREATE TABLE IF NOT EXISTS news_articles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source_name VARCHAR(255),
    title TEXT,
    description TEXT,
    content TEXT,
    published_at DATETIME,
    fetched_from VARCHAR(50)
);


Update credentials in main.py and news_collector.py (default: host="localhost", user="root", password="qwerty00").


Configure API Keys:

Obtain keys for Groq and news APIs (see API Keys).
Update API_KEYS in news_collector.py.
Provide your Groq API key when prompted or in the extension settings.


Run the Flask Server:
python main.py

Access the server at http://0.0.0.0:5000.


Chrome Extension Setup

Load the Extension:

Open Chrome and navigate to chrome://extensions/.
Enable Developer mode (top-right).
Click Load unpacked and select the truthshield directory.


Configure Settings:

Open the TruthShield popup via the extension icon.
In the Settings tab, add your Groq API key and adjust preferences.




🛠️ Usage
Backend Usage

Train Models:Train new models from scratch:
python main.py --train

Use --force-retrain to include all examples:
python main.py --force-retrain


Analyze News:Analyze a headline directly:
python main.py --news "NASA Discovers Alien Life on Mars" --api-key YOUR_GROQ_API_KEY


Fetch News:Collect articles from all APIs:
python news_collector.py --fetch

Or from a specific API:
python news_collector.py --fetch --api newsapi


API Endpoints:

POST /analyze: Analyze news text.curl -X POST http://localhost:5000/analyze \
     -H "Authorization: Bearer YOUR_GROQ_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"news_text": "NASA Discovers Alien Life on Mars", "auto_ai_analysis": true}'


POST /ask: Ask a question about news.curl -X POST http://localhost:5000/ask \
     -H "Authorization: Bearer YOUR_GROQ_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"news_text": "NASA Discovers Alien Life on Mars", "question": "What evidence supports this claim?"}'




Test Groq API:Verify API connectivity:
python main.py --test-groq --api-key YOUR_GROQ_API_KEY



Chrome Extension Usage

Analyze Text via Context Menu:

Select text on any webpage.
Right-click and choose Analyze with TruthShield.
View results in the popup.


Manual Analysis:

Open the popup and paste text in the Analyze tab.
Click Analyze Text.


Explore Features:

Results: View verdict, confidence, and detailed analysis.
Ask Questions: Use Ask a Question for AI-driven answers.
History: Save and review past analyses.
Settings: Adjust confidence thresholds and enable auto AI analysis.




📁 Project Structure
truthshield/
├── True/                     # 📂 Real news data (txt, csv)
├── Fake/                     # 📂 Fake news data (txt, csv)
├── models/                   # 📂 Trained ML models and vectorizer
├── icons/                    # 🖼️ Extension icons (16/48/128px)
├── main.py                   # 🧠 Flask backend and ML logic
├── news_collector.py         # 🌐 News fetching script
├── background.js             # ⚙️ Extension background script
├── content.js                # 📜 Extension content script
├── popup.html                # 🖥️ Extension popup UI
├── popup.css                 # 🎨 Extension popup styles
├── popup.js                  # 🛠️ Extension popup logic
├── manifest.json             # 📋 Extension manifest
└── README.md                 # 📖 Project documentation


🔍 Technical Details
Machine Learning

Models:
Random Forest: 100 estimators, max depth 20, balanced weights.
Logistic Regression: C=1.0, max_iter=1000, balanced weights.


Features: TF-IDF vectorization (stop words, max_df=0.7, min_df=3, ngrams 1-2).
Data: Loads from True/Fake folders, CSV files, MySQL, or dummy data.
Retraining: Supports incremental updates with AI or user feedback.

AI Analysis

Groq API: Uses meta-llama/llama-4-scout-17b-16e-instruct model.
Fallback: Rule-based analysis checks for sensationalist words, punctuation, and clickbait phrases.

News Collection

APIs: NewsAPI, GNews, MediaStack, NewsData.io, Google RSS, CurrentsAPI, NYTimes.
Storage: MySQL database with unified schema.
Processing: Cleans text and normalizes date formats.

Chrome Extension

Manifest: V3 with context menu and popup.
Storage: Uses Chrome storage API for history and settings.
UI: Tabbed interface for analysis, history, and settings.

Flask Backend

Endpoints: /analyze (news classification), /ask (follow-up questions).
Responses: JSON with verdict, confidence, and analysis details.


🧩 Dependencies

Python:
    flask
    requests
    mysql-connector-python
    pandas
    numpy
    scikit-learn
    pickle5
    feedparser


Chrome Extension: Native Chrome APIs only.

Install Python dependencies:
pip install flask requests mysql-connector-python pandas numpy scikit-learn pickle5 feedparser


🗄️ Database Setup

Install MySQL:

Ubuntu: sudo apt-get install mysql-server
macOS: brew install mysql
Windows: MySQL Installer


Create Database:Use the SQL script in Installation.

Update Credentials:Modify main.py and news_collector.py if credentials differ.



🔑 API Keys
Obtain keys for:

Groq: groq.com
News APIs:
NewsAPI: newsapi.org
GNews: gnews.io
MediaStack: mediastack.com
NewsData.io: newsdata.io
CurrentsAPI: currentsapi.services
NYTimes: developer.nytimes.com



Update API_KEYS in news_collector.py:
API_KEYS = {
    'newsapi': 'YOUR_NEWSAPI_KEY',
    'gnews': 'YOUR_GNEWS_KEY',
    'mediastack': 'YOUR_MEDIASTACK_KEY',
    'newsdata': 'YOUR_NEWSDATA_KEY',
    'currents': 'YOUR_CURRENTS_KEY',
    'nytimes': 'YOUR_NYTIMES_KEY',
}


🤝 Contributing
We welcome contributions! To get started:

Fork the repository.
Create a branch: git checkout -b feature-name.
Commit changes: git commit -m "Add feature-name".
Push: git push origin feature-name.
Open a pull request with tests and updated documentation.


📜 License
This project is licensed under the MIT License.

TruthShield © 2025 | Built to fight misinformation with technology. 🌍
