import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("GrammarandProductReviews - GrammarandProductReviews.csv")
df = df[['name','brand','categories','reviews.text','reviews.rating']]
df.rename(columns={
    'reviews.text':'review',
    'reviews.rating':'rating'
}, inplace=True)
df.head()

df.dropna(subset=['review'], inplace=True)
df['brand'] = df['brand'].fillna('Unknown').str.strip().str.lower()
df['categories'] = df['categories'].fillna('Unknown')
df['review'] = df['review'].str.lower()

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)
df['clean_review'] = df['review'].apply(clean_text)

def rating_sentiment(r):
    if r >= 4:
        return "Positive"
    elif r == 3:
        return "Neutral"
    else:
        return "Negative"
df['rating_sentiment'] = df['rating'].apply(rating_sentiment)

df['sentiment_score'] = df['clean_review'].apply(
    lambda x: TextBlob(x).sentiment.polarity
)

def text_sentiment(score):
    if score > 0:
        return "Positive"
    elif score == 0:
        return "Neutral"
    else:
        return "Negative"
df['text_sentiment'] = df['sentiment_score'].apply(text_sentiment)

sns.countplot(x='text_sentiment', data=df)
plt.title("Sentiment Distribution")
plt.show()

brand_sentiment = df.groupby('brand')['sentiment_score'].mean().sort_values(ascending=False)
print("Top Loved Brands:")
print(brand_sentiment.head(10))

category_sentiment = df.groupby('categories')['sentiment_score'].mean().sort_values(ascending=False)

print("Top Categories:")
print(category_sentiment.head(10))

X = df['clean_review']
y = df['rating_sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_test, y_pred))
product_summary = (
    df.groupby('name')
    .agg(
        avg_rating=('rating','mean'),
        review_count=('rating','count'),
        avg_sentiment=('sentiment_score','mean')
    )
)
positive_ratio = (
    df[df['text_sentiment']=="Positive"]
    .groupby('name')
    .size()
    /
    df.groupby('name').size()
)

product_summary['positive_ratio'] = positive_ratio.fillna(0)
product_summary['smart_score'] = (
    product_summary['avg_rating'] * 0.4 +
    product_summary['avg_sentiment'] * 0.3 +
    product_summary['positive_ratio'] * 0.2 +
    np.log1p(product_summary['review_count']) * 0.1
)
smart_products = product_summary.sort_values(
    by='smart_score',
    ascending=False
)

smart_products.head(10)
def recommend_top(n=5):
    return smart_products.head(n)

def recommend_by_brand(brand_name, n=5):
    return df[df['brand'].str.contains(brand_name, na=False)] \
        .groupby('name')['rating'].mean() \
        .sort_values(ascending=False).head(n)

def recommend_by_category(category, n=5):
    return df[df['categories'].str.contains(category, case=False, na=False)] \
        .groupby('name')['rating'].mean() \
        .sort_values(ascending=False).head(n)
recommend_top()
recommend_by_brand("logitech")
recommend_by_category("headphones")

df['mismatch'] = df['rating_sentiment'] != df['text_sentiment']


mismatch_percent = round(df['mismatch'].mean() * 100, 2)

print(f"Mismatch Reviews: {mismatch_percent}%")