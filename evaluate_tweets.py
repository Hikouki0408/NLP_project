import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a mapping of sentiment labels to numerical values
sentiment_map = {'negative': 0, 'neutral': 2, 'positive': 4}

# Define a custom dataset class
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, tfidf_vectorizer):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.tfidf_vectorizer = tfidf_vectorizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        sentiment = self.data.iloc[idx]['sentiment']
        tokens = self.tokenizer(text)
        tfidf_features = self.tfidf_vectorizer.transform([" ".join(tokens)]).toarray()[0]

        # Map sentiment labels to numerical values here
        sentiment_label = sentiment_map[sentiment]

        return torch.FloatTensor(tfidf_features), torch.LongTensor([sentiment_label])

# Define a simple sentiment analysis model
class SentimentModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SentimentModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

# Define a tokenizer (you can customize this further)
def tokenize_text(text):
    # Check if the input is a string
    if not isinstance(text, str):
        return []

    # Tokenize text and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stopwords.words('english')]
    return tokens

def map_sentiment(value):
    if value == '0':
        return 'negative'
    elif value == '2':
        return 'neutral'
    elif value == '4':
        return 'positive'
    else:
        return 'unknown'

def read_text_from_csv(file_path, max_instances=1000):
    try:
        text_list = []
        with open(file_path, 'r', encoding='latin-1') as csv_file:
            csv_reader = csv.reader(csv_file)
            for idx, row in enumerate(csv_reader, start=1):
                if len(row) >= 6:  # Checking if column F exists in the row
                    sentiment_label = row[0]
                    sentiment = map_sentiment(sentiment_label)
                    text = row[5]  # Column F corresponds to index 5
                    text_list.append((sentiment, text))
                else:
                    text_list.append(('unknown', ''))  # Append unknown sentiment and empty text if columns don't exist

                if idx >= max_instances:
                    break  # Stop reading after reaching max_instances

        return text_list
    except Exception as e:
        print("An error occurred:", e)
        return None

def preprocess_data(data):
    # Tokenize text and remove stopwords
    tokenized_data = [(sentiment, tokenize_text(text)) for sentiment, text in data]

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, tokenizer=lambda x: x, preprocessor=lambda x: x)

    # Fit the vectorizer on the tokenized text data
    tfidf_vectorizer.fit([text for sentiment, text in tokenized_data])

    # Map sentiment labels to numerical values
    labels = [sentiment_map[sentiment] for sentiment, _ in data]

    return tokenized_data, labels, tfidf_vectorizer

def train_model(train_loader, model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.squeeze())  # Use squeeze to remove extra dimension
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def main():
    file_path = 'datasets/dataset_tweet.csv'  # Update this with the actual file path
    max_instances = 1000  # Specify the maximum number of instances to read

    text_list = read_text_from_csv(file_path, max_instances)

    if text_list:
        print("Sentiment and Text from columns A and F:")
        for idx, (sentiment, text) in enumerate(text_list, start=1):
            print(f"{idx}: {sentiment.capitalize()}: {text}")
    else:
        print("Failed to read text from the CSV file.")

    # Preprocess the data
    tokenized_data, labels, tfidf_vectorizer = preprocess_data(text_list)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tokenized_data, labels, test_size=0.2, random_state=42)

    # Create data loaders
    train_dataset = SentimentDataset(pd.DataFrame(X_train, columns=['sentiment', 'text']), tokenize_text, tfidf_vectorizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define model parameters
    input_size = len(tfidf_vectorizer.get_feature_names_out())
    num_classes = len(set(labels))

    # Initialize the model and optimizer
    model = SentimentModel(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(train_loader, model, criterion, optimizer, num_epochs=10)

    # Continue with evaluation and inference as needed

if __name__ == "__main__":
    main()
