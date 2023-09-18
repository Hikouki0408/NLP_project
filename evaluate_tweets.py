import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Define a mapping of sentiment labels to numerical values
sentiment_map = {'negative': 0, 'neutral': 2, 'positive': 4}

# Define a custom dataset class
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, tfidf_vectorizer):
        self.data = dataframe  # Store the DataFrame containing the data
        self.tokenizer = tokenizer  # Store the tokenizer function
        self.tfidf_vectorizer = tfidf_vectorizer  # Store the TF-IDF vectorizer

    def __len__(self):
        return len(self.data)  # Return the number of data instances

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']  # Get the text from the DataFrame
        sentiment = self.data.iloc[idx]['sentiment']  # Get the sentiment label from the DataFrame
        tokens = self.tokenizer(text)  # Tokenize the text
        tfidf_features = self.tfidf_vectorizer.transform([" ".join(tokens)]).toarray()[0]  # Compute TF-IDF features

        # Map sentiment labels to numerical values here
        sentiment_label = sentiment_map[sentiment]  # Convert sentiment label to numerical value

        return torch.FloatTensor(tfidf_features), torch.LongTensor([sentiment_label])  # Return TF-IDF features and sentiment label

# Define a simple LSTM-based sentiment analysis model
class SentimentModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SentimentModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        return out

# Define a tokenizer (you can customize this further)
def tokenize_text(text):
    # Check if the input is a string
    if not isinstance(text, str):
        return []

    # Tokenize text and remove stopwords
    tokens = word_tokenize(text)  # Tokenize the input text
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stopwords.words('english')]  # Preprocess tokens
    return tokens  # Return the tokenized and preprocessed tokens

# Define a function to map numerical sentiment values to labels
def map_sentiment(value):
    if value == '0':
        return 'negative'
    elif value == '2':
        return 'neutral'
    elif value == '4':
        return 'positive'
    else:
        return 'unknown'

# Define a function to read text data from a CSV file
def read_text_from_csv(file_path):
    try:
        text_list = []
        with open(file_path, 'r', encoding='latin-1') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) >= 6:  # Checking if column F exists in the row
                    sentiment_label = row[0]
                    sentiment = map_sentiment(sentiment_label)
                    text = row[5]  # Column F corresponds to index 5
                    text_list.append((sentiment, text))
                else:
                    text_list.append(('unknown', ''))  # Append unknown sentiment and empty text if columns don't exist

        return text_list
    except Exception as e:
        print("An error occurred:", e)
        return None

# Define a function to preprocess the data
def preprocess_data(data):
    # Tokenize text and remove stopwords
    tokenized_data = [(sentiment, tokenize_text(text)) for sentiment, text in data]

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, tokenizer=lambda x: x, preprocessor=lambda x: x)

    # Fit the vectorizer on the tokenized text data
    tfidf_vectorizer.fit([" ".join(tokens) for _, tokens in tokenized_data])

    # Map sentiment labels to numerical values
    labels = [sentiment_map[sentiment] for sentiment, _ in data]

    return tokenized_data, labels, tfidf_vectorizer

# Define a function for model training
def train_model(train_loader, model, criterion, optimizer, num_epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass through the model
            loss = criterion(outputs, labels.squeeze())  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

# Define a function for model evaluation
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# Define the main function
def main():
    file_path = 'datasets/dataset_tweet.csv'  # Update this with the actual file path

    text_list = read_text_from_csv(file_path)  # Read text data from CSV

    if text_list:
        print("Sentiment and Text from columns A and F:")
        for idx, (sentiment, text) in enumerate(text_list, start=1):
            print(f"{idx}: {sentiment.capitalize()}: {text}")
    else:
        print("Failed to read text from the CSV file.")

    # Preprocess the data
    labels, tfidf_vectorizer = preprocess_data(text_list)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tfidf_vectorizer.transform([" ".join(tokens) for _, tokens in text_list]), labels, test_size=0.2, random_state=42)

    # Create data loaders for batch processing
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train.toarray()), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test.toarray()), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define model parameters
    input_size = X_train.shape[1]
    hidden_size = 128
    num_layers = 2
    num_classes = len(set(labels))

    # Initialize the model and optimizer
    model = SentimentModel(input_size, hidden_size, num_layers, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(train_loader, model, criterion, optimizer, num_epochs=10)

    # Evaluate the model
    print("Model Evaluation:")
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
