# **📊 Prodigy InfoTech Internship – Task 4**
# **Social Media Sentiment Analysis & Visualization**

---

## **📌 Task Objective**

Analyze and visualize sentiment patterns in social media data to understand public opinion and attitudes towards specific brands using natural language processing (NLP) techniques.

---

## **📁 Dataset**

The dataset used is a Twitter Airline Sentiment Dataset, containing tweets about major U.S. airlines.
Download Airline Tweet Sentiment Dataset (Kaggle): https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

### Key Columns:
- **text – Tweet content**
- **airline – Airline name**
- **airline_sentiment – Sentiment label (positive, neutral, negative)**
- **confidence scores – Sentiment confidence**
- **tweet metadata – Time, user, location**
- **Unnecessary columns were removed for cleaner analysis.**

 ---
 
## **🛠 Tools & Libraries**
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- TextBlob (for sentiment polarity)

 ---
 
## **📊 Exploratory Data Analysis & Visualizations**

### 1. Sentiment Distribution
Bar chart showing overall sentiment across all tweets.

**Insights:**
- Majority of tweets are negative.
- Neutral tweets come next.
- Positive tweets are the least.

---
  
### 2. Sentiment by Airline
Countplot showing sentiment distribution for each airline.

**Insights:**
- Some airlines receive significantly more negative feedback.
- Customer dissatisfaction is uneven across brands.

---

### 3. Tweet Length Distribution
Histogram of tweet character lengths.

**Insights:**
- Most tweets are short.
- Extremely long tweets are rare.

---

### 4. Sentiment Polarity Distribution
Histogram of polarity scores generated using TextBlob.

**Insights:**
- Polarity ranges from -1 to +1.
- Most tweets cluster around negative values.
- Positive tweets have higher polarity scores.

---

### 5. Pie Chart of Sentiment Proportion
Shows percentage share of each sentiment category.

---

## **📂 Project Structure**
Prodigy_DS_Task4/ │ ├── Tweets.csv ├── task4_sentiment.ipynb └── task4_sentiment.py

---

## **▶ How to Run**

### **1. Install dependencies**
pip install pandas numpy matplotlib seaborn textblob

### **2. Run the script**
- python task4_sentiment.py (or)
- open the notebook task4_sentiment.ipynb and run all cells.

---

## **📈 Conclusion**
- This project demonstrates how sentiment analysis helps businesses understand customer opinions.
- Negative sentiment dominates airline tweets, highlighting service improvement areas and customer experience challenges.

---

## **✨ Author**

**Sahana M**

**Prodigy Infotech**
**Data Science Intern**
