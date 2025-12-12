# ============================================================
# ADVANCED SENTIMENT + EMOTION ANALYSIS PROJECT
# ============================================================

import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ============================================================
# 1. LOAD DATA (SAFE WAY)
# ============================================================

# Read reviews.csv line by line
with open("reviews.csv", "r", encoding="utf-8") as f:
    reviews = [line.strip() for line in f if line.strip()]

# Create DataFrame
df = pd.DataFrame(reviews, columns=["review"])

print("\n================ DATA LOADED ================\n")
print(df.head())

# ============================================================
# 2. CLEAN TEXT
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_review'] = df['review'].apply(clean_text)

print("\n================ CLEANED TEXT ================\n")
print(df[['review', 'clean_review']].head())

# ============================================================
# 3. SENTIMENT ANALYSIS (VADER)
# ============================================================

analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = analyzer.polarity_scores(text)
    comp = score['compound']
    if comp >= 0.05:
        return "Positive"
    elif comp <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['vader_sentiment'] = df['clean_review'].apply(vader_sentiment)

# ============================================================
# 4. TEXTBLOB POLARITY (DOUBLE CHECK)
# ============================================================

def blob_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df['blob_sentiment'] = df['clean_review'].apply(blob_sentiment)

# ============================================================
# 5. FINAL SENTIMENT (COMBINED)
# ============================================================

def combine_sentiment(vader, blob):
    if vader == blob:
        return vader
    else:
        return vader   # majority vote, can be customized

df['final_sentiment'] = df.apply(lambda x: combine_sentiment(x['vader_sentiment'], x['blob_sentiment']), axis=1)

print("\n================ FINAL SENTIMENT ================\n")
print(df[['review','final_sentiment']].head())

# ============================================================
# 6. EMOTION DETECTION (LEXICON BASED)
# ============================================================

emotion_words = {
    "joy": ["happy","love","great","amazing","good","wonderful"],
    "anger": ["angry","hate","worst","terrible","bad"],
    "sadness": ["sad","unhappy","disappointed","pain"],
    "fear": ["scared","fear","afraid","worried"],
    "surprise": ["wow","unexpected","shocked","surprising"]
}

def detect_emotion(text):
    words = text.split()
    emotion_score = {emotion:0 for emotion in emotion_words}

    for word in words:
        for emo, emo_list in emotion_words.items():
            if word in emo_list:
                emotion_score[emo] += 1

    return max(emotion_score, key=emotion_score.get)

df["emotion"] = df["clean_review"].apply(detect_emotion)

print("\n================ EMOTION DETECTED ================\n")
print(df[['review','emotion']].head())

# ============================================================
# 7. WORDCLOUDS (POSITIVE / NEGATIVE)
# ============================================================

positive_text = " ".join(df[df['final_sentiment']=="Positive"]["clean_review"])
negative_text = " ".join(df[df['final_sentiment']=="Negative"]["clean_review"])

pos_wc = WordCloud(width=800, height=400, background_color="white").generate(positive_text)
neg_wc = WordCloud(width=800, height=400, background_color="white").generate(negative_text)

plt.figure(figsize=(10,5))
plt.imshow(pos_wc)
plt.axis("off")
plt.title("Positive WordCloud")
plt.savefig("positive_wordcloud.png")
plt.show()

plt.figure(figsize=(10,5))
plt.imshow(neg_wc)
plt.axis("off")
plt.title("Negative WordCloud")
plt.savefig("negative_wordcloud.png")
plt.show()

# ============================================================
# 8. VISUALIZATIONS
# ============================================================

# PIE CHART
sent_count = df['final_sentiment'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(sent_count, labels=sent_count.index, autopct="%.1f%%")
plt.title("Sentiment Distribution")
plt.savefig("sentiment_pie.png")
plt.show()

# BAR CHART
emo_count = df['emotion'].value_counts()
plt.figure(figsize=(8,4))
sns.barplot(x=emo_count.index, y=emo_count.values)
plt.title("Emotion Count")
plt.savefig("emotion_bar.png")
plt.show()

# ============================================================
# 9. SAVE RESULTS
# ============================================================

df.to_csv("sentiment_results_advanced.csv", index=False)

print("\n========== PROJECT COMPLETED SUCCESSFULLY 🔥😎 ==========")
print("Results saved as: sentiment_results_advanced.csv")
print("All charts saved as PNG files!")
