# Machine-Generated-Vs-Human-Generated-Text-classification-Model
This project focuses on distinguishing between human-written and machine-generated tweets using advanced machine learning techniques. It aims to combat misinformation and enhance trustworthiness on social media platforms like Twitter by automating the detection process.


Key Features

Dataset:
Combined 30,000 tweets from three topics:
FIFA World Cup 2022
US Election 2020
Game of Thrones Season 8
Each dataset contains 10,000 rows and 10 features, including text, token length, emoji count, profile count, URL count, and human/machine labels.
Preprocessing Pipeline:
Text cleaning (removed punctuation, emojis, profiles, and URLs).
Tokenization and lemmatization using SpaCy.
Language filtering using FastText to isolate English tweets.
Maintained stopwords for improved model accuracy.
Random sampling to ensure a balanced representation across datasets.
Modeling Approach:
Implemented Random Forest (RF), Decision Tree (DT), Support Vector Machine (SVM), and Logistic Regression (LR) models.
K-fold Cross-Validation (k=5) applied to validate model performance.
Results:
Highest Accuracy:
Game of Thrones dataset (RF Model): 99.98%
FIFA dataset (LR Model): 76.90%
Key Observations:
Human-generated tweets had more diverse sentiment and linguistic complexity.
Machine-generated tweets showed neutral tones and simpler readability.
Sentiment & Readability Analysis:
Sentiment scored using TextBlob and readability analyzed via TextStat (Flesch-Kincaid grade level).
Statistical tests revealed significant differences in sentiment and readability across datasets.
Personality Traits Analysis:
Tokenized data processed through BERTForSequenceClassification to identify traits like Extraversion and Neuroticism.
Machine-generated tweets displayed normalized traits, while human tweets showed natural variability.
Technical Highlights

Languages & Tools: Python, SpaCy, FastText, OpenAI GPT API, BERT, Scikit-learn.
Machine Learning Techniques: Classification models (RF, DT, SVM, LR), K-fold cross-validation.
Natural Language Processing (NLP): Sentiment analysis, readability assessment, personality trait prediction.
Data Visualization: Token distribution, profile/emoji counts, and sentiment trends visualized for insights.
Key Metrics

Dataset	Model	Feature Set	Accuracy (%)
Game of Thrones	Random Forest (RF)	Clean Tweets	99.98
FIFA	Logistic Regression (LR)	Clean Tweets + Emoji/Profile Count	76.90
US Election 2020	Logistic Regression (LR)	Clean Tweets	72.38
Impact

Enables real-time identification of machine-generated content, aiding in reducing misinformation.
Promotes trustworthiness on social platforms by analyzing linguistic patterns.
Future Work

Incorporate deep learning techniques to improve detection accuracy.
Pool datasets to create a generalized detection model.
Extend the project to other domains like news articles or reviews.
Repository Structure

data/: Preprocessed datasets.
notebooks/: Jupyter notebooks for exploratory data analysis (EDA), modeling, and visualization.
models/: Trained machine learning models.
scripts/: Python scripts for data cleaning, feature extraction, and analysis.
results/: Model evaluation metrics and visualizations.
