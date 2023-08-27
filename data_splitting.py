import csv
from sklearn.model_selection import train_test_split

def map_sentiment(value):
    if value == '0':
        return 'negative'
    elif value == '2':
        return 'neutral'
    elif value == '4':
        return 'positive'
    else:
        return 'unknown'

def read_text_from_csv(file_path):
    try:
        text_list = []
        with open(file_path, 'r', encoding='latin-1') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) >= 6:  # Checking if column F exists in the row
                    sentiment = map_sentiment(row[0])  # Column A corresponds to index 0
                    text = row[5]  # Column F corresponds to index 5
                    text_list.append((sentiment, text))
                else:
                    text_list.append(('unknown', ''))  # Append unknown sentiment and empty text if columns don't exist
        return text_list
    except Exception as e:
        print("An error occurred:", e)
        return None

def main():
    file_path = 'datasets/dataset_tweet.csv'  # Update this with the actual file path

    text_list = read_text_from_csv(file_path)

    if text_list:
        # Split the data into training, validation, and test sets
        train_data, test_data = train_test_split(text_list, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

        print("Number of instances in:")
        print(f"Training set: {len(train_data)}")
        print(f"Validation set: {len(val_data)}")
        print(f"Test set: {len(test_data)}")

        # You can now use train_data, val_data, and test_data in the subsequent steps

    else:
        print("Failed to read text from the CSV file.")

if __name__ == "__main__":
    main()

    """
    Training Set: There are 1,152,000 instances (tweets) in the training set. This set is used to train your sentiment analysis model. 
    It's the largest set and contains the most data for training the model's parameters.

    Validation Set: There are 128,000 instances (tweets) in the validation set. This set is used for tuning hyperparameters and 
    monitoring the model's performance during training. It helps prevent overfitting by providing an independent dataset for assessing how well the model generalizes to new data.

    Test Set: There are 320,000 instances (tweets) in the test set. This set is used to evaluate the final performance of your trained model. 
    After your model is trained and its hyperparameters are tuned using the training and validation sets, 
    you use the test set to assess how well the model performs on unseen data. This helps you estimate how well your model will perform in real-world scenarios.
    """