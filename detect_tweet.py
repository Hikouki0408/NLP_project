import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification  # Import BERT-related libraries here
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Download NLTK stopwords if you haven't already
nltk.download('stopwords')
nltk.download('punkt')

# Define the global tokenizer and label_encoder
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
label_encoder = LabelEncoder()

# Import BERT-related libraries
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Function to map numeric sentiment labels to human-readable labels
def map_sentiment(value):
    if value == '0':
        return 'negative'
    elif value == '2':
        return 'neutral'
    elif value == '4':
        return 'positive'
    else:
        return 'unknown'

# Function to preprocess text data
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

# Function to read text from a CSV file and preprocess it
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

# Function to fine-tune a BERT model for sentiment analysis
def fine_tune_bert(X_train, y_train):
    # Load pre-trained BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Tokenize and convert text data to PyTorch tensors
    X_train_encoded = tokenizer(X_train, padding=True, truncation=True, return_tensors='pt', max_length=64)

    # Encode labels (negative, neutral, positive)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)  # Define label_encoder here

    # Create DataLoader for batch processing
    train_data = TensorDataset(X_train_encoded.input_ids, X_train_encoded.attention_mask, torch.tensor(y_train_encoded))
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

    # Define optimizer and loss function (you may need to fine-tune hyperparameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop (you may need to adjust the number of epochs and other hyperparameters)
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            output = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(output.logits, labels)
            loss.backward()
            optimizer.step()

    # Return the fine-tuned model
    return model


# Function to evaluate a BERT model on test data
def evaluate_bert(model, X_test, y_test):
    # Tokenize and convert test data to PyTorch tensors
    X_test_encoded = tokenizer(X_test, padding=True, truncation=True, return_tensors='pt', max_length=64)

    # Encode test labels
    y_test_encoded = label_encoder.transform(y_test)

    # Create DataLoader for batch processing
    test_data = TensorDataset(X_test_encoded.input_ids, X_test_encoded.attention_mask, torch.tensor(y_test_encoded))
    test_dataloader = DataLoader(test_data, batch_size=16)

    # Evaluation
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, _ = batch
            output = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(output.logits, dim=1)
            all_predictions.extend(predictions.tolist())

    # Inverse transform labels for the classification report
    y_test_decoded = label_encoder.inverse_transform(y_test_encoded)
    y_pred_decoded = label_encoder.inverse_transform(all_predictions)

    # Get classification report
    report = classification_report(y_test_decoded, y_pred_decoded, target_names=['negative', 'neutral', 'positive'])

    return report

# Main function
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

        # Fine-tune BERT model
        bert_model = fine_tune_bert(X_train, y_train)

        # Evaluate BERT model
        report = evaluate_bert(bert_model, X_test, y_test)

        print("BERT Model Results (fine-tuning may be needed):")
        print("Classification Report:")
        print(report)
    else:
        print("Failed to read text from the CSV file.")

if __name__ == "__main__":
    main()
