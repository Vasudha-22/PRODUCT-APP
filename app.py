

import streamlit as st
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Product Intelligence System", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("GrammarandProductReviews - GrammarandProductReviews.csv")

    # select relevant columns
    df = df[['name','brand','categories','reviews.text','reviews.rating']]

    # rename for simplicity
    df.columns = ['product_name','brand','category','review_text','rating']

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # clean text
    df['review_text'] = df['review_text'].astype(str).str.lower()
    df['review_text'] = df['review_text'].apply(lambda x: re.sub(r'[^a-zA-Z ]','',x))

    # extract primary category
    df['category'] = df['category'].apply(lambda x: x.split(',')[0])

    # sentiment labels
    def sentiment_label(r):
        if r >= 4:
            return "Positive"
        elif r == 3:
            return "Neutral"
        else:
            return "Negative"

    df['Sentiment'] = df['rating'].apply(sentiment_label)

    # score for recommendation
    df['score'] = df['rating'] + df['Sentiment'].map({
        "Positive":2, "Neutral":1, "Negative":0
    })

    return df

df = load_data()


@st.cache_resource
def train_model(data):
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(data['review_text'])
    y = data['Sentiment']

    model = LogisticRegression()
    model.fit(X, y)

    return model, vectorizer

model, vectorizer = train_model(df)

st.title("🛍️ Product Review Intelligence & Recommendation System")
page = st.sidebar.radio(
    "Navigate",
    ["Dataset Overview", "Top Recommendations", "Brand & Product Explorer"]
)

#1
if page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Reviews", len(df))
    col2.metric("Total Brands", df['brand'].nunique())
    col3.metric("Total Categories", df['category'].nunique())

    st.subheader("Sentiment Distribution")
    st.bar_chart(df['Sentiment'].value_counts())

    st.subheader("Top Brands")
    st.bar_chart(df['brand'].value_counts().head(10))

    st.subheader("Top Categories")
    st.bar_chart(df['category'].value_counts().head(10))
    @st.cache_resource
    def train_model(data):
        vectorizer = TfidfVectorizer(max_features=3000)
        X = vectorizer.fit_transform(data['review_text'])
        y = data['Sentiment']

        model = LogisticRegression()
        model.fit(X, y)

        accuracy = model.score(X, y)

        return model, vectorizer, accuracy

    model, vectorizer, accuracy = train_model(df)
    st.subheader("🤖 Model Accuracy")
    st.write(round(accuracy*100,2), "%")

    st.subheader("🔥 Most Reviewed Products")
    st.bar_chart(df['product_name'].value_counts().head(10))

#2
elif page == "Top Recommendations":

    st.title("⭐ Top Recommended Products")

    st.write("### Best Products Overall")
    top_products = df.sort_values(by='score', ascending=False)

    st.dataframe(
        top_products[['product_name','brand','rating','Sentiment']]
        .drop_duplicates()
        .head(10)
    )

    st.write("### Category Wise Recommendation")
    cat = st.selectbox("Select Category", df['category'].unique())

    filtered = top_products[top_products['category']==cat]

    st.table(
        filtered[['product_name','brand','rating','Sentiment']]
        .drop_duplicates()
        .head(5)
    )

#3
elif page == "Brand & Product Explorer":

    st.title("🔍 Brand & Product Explorer")

    brand = st.selectbox("Select Brand", sorted(df['brand'].unique()))

    products = df[df['brand']==brand]['product_name'].unique()

    product = st.selectbox("Select Product", products)

    product_data = df[df['product_name']==product]

    st.subheader("⭐ Average Rating")
    st.write(round(product_data['rating'].mean(),2))

    st.subheader("📊 Sentiment Summary")
    st.bar_chart(product_data['Sentiment'].value_counts())

    st.subheader("📝 Sample Reviews")
    st.write(product_data['review_text'].head(5))

    st.subheader("🔥 Similar Recommended Products")

    recommended = df[
        (df['brand']==brand) &
        (df['product_name']!=product)
    ].sort_values(by='score', ascending=False)

    st.table(
        recommended[['product_name','rating','Sentiment']]
        .drop_duplicates()
        .head(5)
    )
