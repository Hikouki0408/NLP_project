import urllib.request
from html.parser import HTMLParser
import re
import spacy
from selenium import webdriver


class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside_head_tag = False
        self.inside_body_tag = False
        self.skip_tag_body = False
        self.select_tag_head = False
        self.extract_script = False
        self.tag_stack = []
        self.data_by_label = {}
        self.nlp = spacy.load("en_core_web_sm") # loads a pre-trained English language model 
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.url_pattern = re.compile(r'^(?:https?://|www\.)\S+\.[a-zA-Z]{2,}(?:[/?#]\S*)?$')
        self.date_patterns = ["\d{1,2}\/\d{1,2}\/\d{4}", "\d{1,2}-\d{1,2}-d{4}"]
        self.count = 0
        self.text = ""
        
    def handle_starttag(self, tag, attrs):
        if tag == "head":                         # Check if the current tag is the opening <head> tag
            self.inside_head_tag = True           # Set the flag to indicate being inside the <head> tag
        if tag == "body":                         # Check if the current tag is the opening <body> tag
            self.inside_body_tag = True           # Set the flag to indicate being inside the <body> tag
        self.tag_stack.append(tag)               # Add the tag to the tag stack for tracking nested tags

        if tag in ["title"]:                     # Check if the current tag is one of the specified tags (e.g., <title>)
            self.select_tag_head = True          # Set the flag to indicate selecting content from the <head> tag (e.g., <title> content)
        if tag in ["script"]:                    # Check if the current tag is one of the specified tags (e.g., <script>)
            self.extract_script = True           # Set the flag to indicate extracting content from the <script> tag

        if tag in ["style", "iframe", "svg", "form", "button", "footer", "nav"]:
            # Check if the current tag is one of the specified tags to filter out in the <body> section
            self.skip_tag_body = True            # Set the flag to skip processing content from this tag in the <body>

    def handle_endtag(self, tag):
        if tag == "head":                           # Check if the end tag is for the <head> element
            self.inside_head_tag = False            # Set 'inside_head_tag' to False since the <head> section ends
        if tag == "body":                           # Check if the end tag is for the <body> element
            self.inside_body_tag = False            # Set 'inside_body_tag' to False since the <body> section ends
        if tag in ["title"]:                        # Check if the end tag is for the <title> element
            self.select_tag_head = False            # Set 'select_tag_head' to False since the <title> section ends
        if tag in ["script"]:                       # Check if the end tag is for the <script> element
            self.extract_script = False             # Set 'extract_script' to False since the <script> section ends
        if tag in ["style", "iframe", "svg", "form", "button", "footer", "nav"]:
            # Check if the end tag is for any of the elements listed (style, iframe, svg, form, button, footer, nav)
            self.skip_tag_body = False              # Set 'skip_tag_body' to False since the corresponding section ends
        self.tag_stack.pop()                        # Remove the last tag from 'tag_stack' since it's the ending tag


    def execute_javascript_code(self, script_code):
        # Set Chrome options for running the browser in a headless mode (without UI)
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")

        # Create a new instance of the Chrome browser with the defined options
        driver = webdriver.Chrome(options=options)

        # Load a blank page in the browser to execute the provided JavaScript code
        driver.get('data:text/html;charset=utf-8,<!DOCTYPE html><html><head></head><body></body></html>')

        try:
            # Execute the provided JavaScript code in the browser
            driver.execute_script(script_code)

            # Extract the dynamically generated text displayed on the browser
            script_extract_text = '''
                var elements = document.querySelectorAll('body > *');
                var texts = "";
                elements.forEach(function(element) {
                    texts += element.textContent + " ";
                });
                return texts.trim();
            '''
            extracted_text = driver.execute_script(script_extract_text)

            # Close the browser after extracting the text
            driver.quit()

            return extracted_text   # Return the extracted text from the dynamically generated content

        except Exception as e:
            # If there's an error during JavaScript execution, print the error message or return an empty string
            return ''

    def handle_data(self, data):
        message = ''                         
        label = "undefined"                  

        if self.inside_head_tag:             # Check if currently inside the <head> tag
            if self.select_tag_head and data.strip():  # If 'select_tag_head' is True and data is not empty, append the stripped data to 'self.text'
                self.text += data.strip() + " "
            elif self.extract_script:        # If 'extract_script' is True, execute JavaScript code and add the extracted content to 'self.text'
                script_code = data.strip()
                extracted_content = self.execute_javascript_code(script_code)
                if extracted_content:
                    self.text += extracted_content + " "

        if self.inside_body_tag:             # Check if currently inside the <body> tag
            if not self.skip_tag_body and data.strip():  # If 'skip_tag_body' is False and data is not empty, proceed with data handling
                tag = self.tag_stack[-1] if self.tag_stack else None   # Get the last tag in the list or None if the list is empty

                if self.extract_script:      # If 'extract_script' is True, execute JavaScript code and update 'data' with the extracted content
                    script_code = data.strip()
                    extracted_content = self.execute_javascript_code(script_code)
                    data = extracted_content

                # Check data for specific patterns (email, URL, non-contextual, or pure text) using regex
                if re.search(self.email_pattern, data) and not any(char.isspace() for char in data.strip()):
                    label = "Email"
                    if label not in self.data_by_label:     # Create an entry for the label in 'self.data_by_label' if it doesn't exist
                        self.data_by_label[label] = []
                    self.data_by_label[label].append(data.strip())   # Add the data to the corresponding label's data list
                    message = f"{label}: [{data.strip()}] (tag: {tag})"  # Construct a message string with the label and data

                elif re.search(self.url_pattern, data) and not any(char.isspace() for char in data.strip()):
                    label = "Website"
                    if label not in self.data_by_label:
                        self.data_by_label[label] = []
                    self.data_by_label[label].append(data.strip())
                    message = f"{label}: [{data.strip()}] (tag: {tag})"

                elif not re.search(r'[a-zA-Z0-9]', data):   # Check if data contains no letters or digits (non-contextual data)
                    label = "Non-contextual"
                    if label not in self.data_by_label:
                        self.data_by_label[label] = []
                    self.data_by_label[label].append(data.strip())
                    message = f"{label}: [{data.strip()}] (tag: {tag})"

                else:   # If none of the above patterns match, consider the data as pure text
                    label = "Puretext"
                    doc = self.nlp(data.strip())   # Apply natural language processing with spaCy
                    named_entities = []
                    for ent in doc.ents:   # Extract named entities like persons, locations, organizations, etc.
                        if ent.label_ in ["PERSON", "GPE", "LANGUAGE", "TIME", "PERCENT", "ORG", "PRODUCT", "EVENT"]:
                            named_entities.append(ent.text)
                        elif ent.label_ == "DATE" and any(char.isdigit() for char in ent.text) and len(ent.text) < 20:
                            named_entities.append(ent.text)
                    
                    if named_entities:   # If named entities are found in the data
                        if label not in self.data_by_label:
                            self.data_by_label[label] = []
                        self.data_by_label[label].append(data.strip())   # Add the data to the corresponding label's data list
                        self.text += data.strip() + " "   # Append the data to 'self.text'
                        message = f"{label}: [{data.strip()}], Labels:[{', '.join([ent.label_ for ent in doc.ents if ent.text in named_entities])}] {', '.join(named_entities)} (tag: {tag})"
                        self.count += 1
                        for ent in doc.ents:
                            if ent.text in named_entities:    # Add named entities to their respective label's data list
                                if ent.label_ not in self.data_by_label:
                                    self.data_by_label[ent.label_] = []
                                self.data_by_label[ent.label_].append(ent.text)
                    else:   # If no named entities are found, consider the data as pure text
                        if label not in self.data_by_label:
                            self.data_by_label[label] = []
                        self.data_by_label[label].append(data.strip())
                        self.text += data.strip() + " "
                        message = f"{label}: [{data.strip()}] (tag: {tag})"
                        self.count += 1

        if message:
            print(message)   # Print the constructed 'message' if it contains any relevant information


    def print_data_by_label(self):
        for label, data in self.data_by_label.items():   # Iterate through the 'self.data_by_label' dictionary
            if label != "Puretext":                     # Exclude the "Puretext" label from printing
                print(f"Data for label '{label}': ", end="")
                for i, datum in enumerate(data):        # Iterate through the data list for each label
                    if i == len(data) - 1:              # Check if it's the last element in the data list
                        print(datum)                    # Print the datum (last element) without a comma at the end
                    else:
                        print(datum, end=", ")          # Print the datum with a comma at the end (not the last element)
    
# 1. https://theluxurytravelexpert.com/2020/12/14/best-hotels-in-the-world 
# 2. https://en.wikipedia.org/wiki/Vrije_Universiteit_Amsterdam
# 3. https://research.ibm.com/blog/utility-toward-useful-quantum
# 4. https://www.hotcars.com/upcoming-cars-worth-waiting-for/#2023-fisker-ocean
# 5. https://www.tudelftcampus.nl/time-to-shake-up-the-pile-driving-industry
# 6. https://hackr.io/blog/what-is-programming
# 7. https://www.engadget.com/best-android-phone-130030805.html?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAMJRC35y42RkEpGFN410RsxpbKvMCO1YlLmbtdzQ8pV8l3LRZ5sWPGJQYf-yEwX7vimbG2qzSJYMbpZ545Hz3cup5XB1qlkb203T1mVAKhmOteZxYDxKoohpFTWRvo-M8MzqByHFRBN4-odKGhQEche2Zb-GXjopL6cIZsxeIuLl
# 8. https://www.amsterdamfoodie.nl/amsterdam-food-guide/indonesian-restaurants-in-amsterdam-rijsttafel
# 9. https://stackoverflow.blog/2023/05/31/ceo-update-paving-the-road-forward-with-ai-and-community-at-the-center
# 10. https://www.euronews.com/travel/2023/02/27/long-queues-and-scams-will-the-new-eu-entry-system-cause-border-chaos

response = urllib.request.urlopen('https://theluxurytravelexpert.com/2020/12/14/best-hotels-in-the-world')
parser = MyHTMLParser()
print("[Parsing HTML file...]")
print()
parser.feed(response.read().decode('utf-8'))
print("[Done parsing HTML file.]")
print()
#parser.print_data_by_label()
print("[Done printing data by label.]")
print(parser.text)


with open('text.txt', 'w') as file:
    file.write(parser.text)

"""
with open('text.txt', 'w') as file:
    file.write(parser.text)
"""