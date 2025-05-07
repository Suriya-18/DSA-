import requests
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Scrape IMDb reviews
def scrape_imdb_reviews(movie_id, max_reviews=50):
    reviews = []
    page = 0

    while len(reviews) < max_reviews:
        url = f"https://www.imdb.com/title/{movie_id}/reviews/_ajax?ref_=undefined&paginationKey={page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        review_containers = soup.find_all("div", class_="text show-more__control")

        if not review_containers:
            break  # No more reviews

        for review in review_containers:
            text = review.get_text().strip()
            reviews.append(text)
            if len(reviews) >= max_reviews:
                break

        page += 1

    return reviews

# Step 2: Analyze sentiment
def analyze_sentiment(reviews):
    sentiment_data = []

    for review in reviews:
        blob = TextBlob(review)
        sentiment = blob.sentiment.polarity
        sentiment_data.append({
            "review": review,
            "sentiment_score": sentiment,
            "sentiment_label": "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
        })

    return pd.DataFrame(sentiment_data)

# Step 3: Visualize results
def visualize_sentiment(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='sentiment_label', palette="coolwarm")
    plt.title("Sentiment Distribution of IMDb Reviews")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.show()

# === Run the program ===
if __name__ == "__main__":
    movie_id = "tt1375666"  # Inception's IMDb ID as an example
    print("Scraping reviews...")
    reviews = scrape_imdb_reviews(movie_id)

    print(f"Collected {len(reviews)} reviews.")
    print("Analyzing sentiment...")
    df = analyze_sentiment(reviews)

    print(df.head())
    visualize_sentiment(df)
