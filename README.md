PROJECT TITLE:- 
🛍️ Product Review Intelligence & Recommendation System

PROJECT OVERVIEW:-
This project is an AI-powered product review analysis and recommendation system that extracts insights from customer reviews and recommends the best products using Natural Language Processing (NLP) and Machine Learning.
It analyzes sentiment, identifies consumer preferences, and provides intelligent product recommendations through an interactive dashboard.

FEATURES:-
1) Sentiment analysis of customer reviews  
2) Positive, Neutral & Negative feedback detection  
3) Brand & category preference insights  
4) Smart product recommendation system  
5) Sentimen​t mismatch detection (fake/misleading reviews)  
6) Machine learning–based sentiment classification  
7) Interactive Streamlit dashboard  
8) Dockerized deployment  

DATASET:-
The project uses the Grammar & Product Reviews dataset, which includes:
- Product names  
- Brands  
- Categories  
- Customer reviews  
- Ratings  

WORKING:-
1) Data Processing
- Text cleaning & normalization  
- Stopword removal & lemmatization  
- Feature extraction using TF-IDF  
2) Sentiment Analysis
- TextBlob polarity scoring  
- Rating-based sentiment labeling  
3) Machine Learning
- Logistic Regression classifier  
- Sentiment prediction & evaluation  
4) Recommendation Logic
Recommendations are generated using:
- Average rating  
- Sentiment score  
- Positive review ratio  
- Review count  
5) Dashboard Insights
- Sentimen​t distribution  
- Top brands & categories  
- Recommended products  
- Brand & product explorer  

PROJECT STRUCTURE:-
product-app/
│
├── app.py                 
├── analysis.py           
├── requirements.txt       
├── Dockerfile             
└── GrammarandProductReviews.csv

RUN LOCALLY:-
1) Install dependencies
pip install -r requirements.txt
2) Run the app
streamlit run app.py
3) Open in browser
[http://localhost:8501](http://localhost:8501)

RUN USING DOCKER:-
1) Build Docker image
docker build -t product-app .
2) Run container
docker run -p 8501:8501 product-app
3) Open the app
[http://localhost:8501](http://localhost:8501)

TECH STACK:-
- Python  
- Streamlit  
- Scikit-learn  
- NLTK  
- TextBlob  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- Docker  

INSIGHTS:-
Sentiment distribution trends  
Most loved brands & categories  
Top-rated & trending products  
Customer satisfaction patterns  
Review sentiment mismatch detection  


