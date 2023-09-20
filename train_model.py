import csv
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Define a mapping of sentiment labels to numerical values
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}

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

# Define a simple sentiment analysis model
class SentimentModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SentimentModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)  # Define a fully connected layer

    def forward(self, x):
        return self.fc(x)  # Forward pass through the fully connected layer

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

# Define a function to read text data from a CSV file up to a specified number of instances
def read_text_from_csv(file_path, max_instances=None):
    try:
        text_list = []
        with open(file_path, 'r', encoding='latin-1') as csv_file:
            csv_reader = csv.reader(csv_file)
            for idx, row in enumerate(csv_reader, start=1):
                if max_instances is not None and idx > max_instances:
                    break  # Stop reading after reaching the specified number of instances
                
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
    tfidf_vectorizer.fit([text for sentiment, text in tokenized_data])

    # Map sentiment labels to numerical values
    labels = [sentiment_map[sentiment] for sentiment, _ in data]

    return tokenized_data, labels, tfidf_vectorizer

# Define a function to train the model
def train_model(train_loader, model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs.float())  # Forward pass through the model
            loss = criterion(outputs, labels.squeeze())  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Define the main function
def main():
    file_path = 'datasets/dataset_tweet.csv'  # Update this with the actual file path
    # Read text data from CSV up to 1,500,000 instances
    text_list = read_text_from_csv(file_path, max_instances=15000)

    if text_list:
        print("Sentiment and Text from columns A and F:")
        for idx, (sentiment, text) in enumerate(text_list, start=1):
            print(f"{idx}: {sentiment.capitalize()}: {text}")
    else:
        print("Failed to read text from the CSV file.")

    # Preprocess the data as before
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

    # Evaluate the model on the test set
    test_dataset = SentimentDataset(pd.DataFrame(X_test, columns=['sentiment', 'text']), tokenize_text, tfidf_vectorizer)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.squeeze().tolist())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1-Score: {f1:.4f}')

if __name__ == "__main__":
    main()
