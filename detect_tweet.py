import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download NLTK stopwords if you haven't already
nltk.download('stopwords')
nltk.download('punkt')

def map_sentiment(value):
    if value == '0':
        return 'negative'
    elif value == '2':
        return 'neutral'
    elif value == '4':
        return 'positive'
    else:
        return 'unknown'

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = ' '.join(word_tokenize(text))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])

    # Stemming (you can choose to include or exclude this step)
    stemmer = PorterStemmer()
    words = text.split()
    text = ' '.join([stemmer.stem(word) for word in words])

    return text

def read_text_from_csv(file_path, max_instances=100):
    try:
        text_list = []
        with open(file_path, 'r', encoding='latin-1') as csv_file:
            csv_reader = csv.reader(csv_file)
            for idx, row in enumerate(csv_reader, start=1):
                if len(row) >= 6:  # Checking if column F exists in the row
                    sentiment = map_sentiment(row[0])  # Column A corresponds to index 0
                    text = row[5]  # Column F corresponds to index 5
                    # Preprocess the text
                    text = preprocess_text(text)
                    text_list.append((sentiment, text))
                else:
                    text_list.append(('unknown', ''))  # Append unknown sentiment and empty text if columns don't exist

                if idx >= max_instances:
                    break  # Stop reading after reaching max_instances
        return text_list
    except Exception as e:
        print("An error occurred:", e)
        return None

def main():
    file_path = 'datasets/dataset_tweet.csv'  # Update this with the actual file path
    max_instances = 100  # Specify the maximum number of instances to read

    text_list = read_text_from_csv(file_path, max_instances)

    if text_list:
        print("Sentiment and Text from columns A and F (preprocessed):")
        for idx, (sentiment, text) in enumerate(text_list, start=1):
            print(f"{sentiment.capitalize()}: {text}")

            # Test print statement to check preprocessing
            original_text = read_text_from_csv(file_path, max_instances)[idx-1][1]
            print(f"Original Text: {original_text}")
            
        # Split data into X (features) and y (labels)
        X = [text for _, text in text_list]
        y = [sentiment for sentiment, _ in text_list]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training set size:", len(X_train))
        print("Testing set size:", len(X_test))

        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        print("TF-IDF Vectorization completed.")
    else:
        print("Failed to read text from the CSV file.")

if __name__ == "__main__":
    main()
