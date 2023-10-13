import csv

def map_sentiment(value):
    if value == '0':
        return 'negative'
    elif value == '2':
        return 'neutral'
    elif value == '4':
        return 'positive'
    else:
        return 'unknown'

def read_text_from_csv(file_path, max_instances=100):
    try:
        text_list = []
        with open(file_path, 'r', encoding='latin-1') as csv_file:
            csv_reader = csv.reader(csv_file)
            for idx, row in enumerate(csv_reader, start=1):
                if len(row) >= 6:  # Checking if column F exists in the row
                    sentiment = map_sentiment(row[0])  # Column A corresponds to index 0
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

def main():
    file_path = 'datasets/dataset_tweet.csv'  # Update this with the actual file path
    max_instances = 1000000  # Specify the maximum number of instances to read

    text_list = read_text_from_csv(file_path, max_instances)

    if text_list:
        print("Sentiment and Text from columns A and F:")
        for idx, (sentiment, text) in enumerate(text_list, start=1):
            print(f"{sentiment.capitalize()}: {text}")
    else:
        print("Failed to read text from the CSV file.")

if __name__ == "__main__":
    main()