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
        if tag == "head":
            self.inside_head_tag = True
        if tag == "body":
            self.inside_body_tag = True
        self.tag_stack.append(tag)
        if tag in ["title"]:
            self.select_tag_head = True
        if tag in ["script"]:
            self.extract_script = True
        if tag in ["style", "iframe", "svg", "form", "button", "footer", "nav"]:
            self.skip_tag_body = True

    def handle_endtag(self, tag):
        if tag == "head":
            self.inside_head_tag = False
        if tag == "body":
            self.inside_body_tag = False
        if tag in ["title"]:
            self.select_tag_head = False
        if tag in ["script"]:
            self.extract_script = False
        if tag in ["style", "iframe", "svg", "form", "button", "footer", "nav"]:
            self.skip_tag_body = False
        self.tag_stack.pop()
    
    def execute_javascript_code(self, script_code):
            # Set Chrome options
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        
        # Create a new instance of the Chrome browser
        driver = webdriver.Chrome(options=options)
        
        driver.get('data:text/html;charset=utf-8,<!DOCTYPE html><html><head></head><body></body></html>')

        try:
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
            # Close the browser
            driver.quit()

            return extracted_text

        except Exception as e:
            # If there's an error, print the error message and return an empty string
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
    
               
with open('index.html', 'r') as file:
    parser = MyHTMLParser()
    print("[Parsing HTML file...]")
    print()
    parser.feed(file.read())
    print("[Done parsing HTML file.]")
    print()
    parser.print_data_by_label()
    print("[Done printing data by label.]")
    print(parser.text)



"""
html_parsing$ python3 test_parse_html.py
[Parsing HTML file...]
Puretext: [KOKI HIROSE], Labels:[PERSON] KOKI HIROSE (tag: a)
Puretext: [Home] (tag: span)
Puretext: [About] (tag: span)
Puretext: [Education] (tag: span)
Puretext: [Projects] (tag: span)
Puretext: [Blogs] (tag: span)
Puretext: [Contact] (tag: span)
Puretext: [© 2023 Koki Hirose], Labels:[DATE] 2023 (tag: p)
Puretext: [English], Labels:[LANGUAGE] English (tag: a)
Non-contextual: [日本語] (tag: a)
Puretext: [Web Developer & Social media content creator], Labels:[ORG] Web Developer & Social (tag: h2)
Puretext: [Multicultural individual from Japan, living in Amsterdam, the Netherlands. Works in web development and filmmaking, driven by tradition and innovation.], Labels:[GPE, GPE, GPE] Japan, Amsterdam, Netherlands (tag: p)
Puretext: [Information about me] (tag: span)
Puretext: [About Me] (tag: h3)
Puretext: [My name is] (tag: h4)
Puretext: [Koki Hirose], Labels:[PERSON] Koki Hirose (tag: span)
Puretext: [Hello, I'm Koki, a web developer and social media content creator with a passion for creating beautiful and engaging digital experiences. As an entry-level developer, I have a foundational knowledge of web development, mobile application development, and programming languages such as C++, Java, and JavaScript.], Labels:[PERSON, PERSON, PERSON, PRODUCT] Koki, C++, Java, JavaScript (tag: p)
Puretext: [I'm excited about combining my technical skills with my creativity to build dynamic and engaging websites and mobile apps. I have a strong eye for design and a passion for creating content that resonates with audiences and drives engagement.] (tag: p)
Puretext: [Although I'm still developing my skills, I'm eager to learn and grow in my role as a web developer and social media content creator. I believe that every project presents an opportunity to improve and I am always looking for ways to refine my skills.] (tag: p)
Puretext: [Degree & Major], Labels:[ORG] Degree & Major (tag: span)
Puretext: [Education] (tag: h3)
Puretext: [- Vrije Universiteit Amsterdam Sep 2020 - Present] (tag: strong)
Puretext: [Bachelor of Science: Computer Science] (tag: br)
Puretext: [Minor: Deep Programming 2022-2023], Labels:[PERSON, DATE] Deep Programming, 2022-2023 (tag: br)
Puretext: [- Study Group Holland ISC Sep 2019 - Aug 2020], Labels:[DATE] 2020 (tag: strong)
Puretext: [International Foundation Year: Science & Engineering], Labels:[ORG, ORG] International Foundation Year:, Science & Engineering (tag: br)
Puretext: [- Tokai University, Japan Apr 2017 - March 2019], Labels:[GPE, DATE] Japan, 2017 - March 2019 (tag: strong)
Puretext: [Bachelor of Human Development: Environmental Science], Labels:[ORG] Bachelor of Human Development: Environmental Science (tag: br)
Puretext: [(completed until the second year)] (tag: br)
Puretext: [My CV is available via LinkedIn.], Labels:[PERSON, GPE] CV, LinkedIn (tag: p)
Website: [https://www.linkedin.com/in/koki-hirose-365a92240] (tag: a)
Puretext: [My Recent Projects] (tag: span)
Puretext: [My Portfolio] (tag: h3)
Puretext: [Portfolios] (tag: h4)
Puretext: [Snake] (tag: h5)
Puretext: [Object Oriented in Scala] (tag: p)
Puretext: [View Project] (tag: a)
Puretext: [Tetris], Labels:[PRODUCT] Tetris (tag: h5)
Puretext: [Object Oriented in Scala] (tag: p)
Puretext: [View Project] (tag: a)
Puretext: [GPX manager] (tag: h5)
Puretext: [Programmed in Java], Labels:[PERSON] Java (tag: p)
Puretext: [View Project] (tag: a)
Puretext: [My Theis] (tag: h5)
Puretext: [Extract text from HTML pages for advanced relation extraction] (tag: p)
Puretext: [View Project] (tag: a)
Puretext: [Work in progress] (tag: h5)
Puretext: [Programmed in Java], Labels:[PERSON] Java (tag: p)
Puretext: [View Project] (tag: a)
Puretext: [Add More...] (tag: a)
Puretext: [Views News], Labels:[ORG] Views News (tag: span)
Puretext: [Latest News], Labels:[ORG] Latest News (tag: h3)
Puretext: [14 April, 2023], Labels:[DATE] 14 April, 2023 (tag: p)
Puretext: [Extract text from HTML pages for advanced relation extraction] (tag: a)
Puretext: [15 April, 2023], Labels:[DATE] 15 April, 2023 (tag: p)
Puretext: [How Chat-GPT is Actually Creating Jobs, Not Stealing Them], Labels:[PERSON, PERSON] Chat-GPT, Actually Creating Jobs (tag: a)
Puretext: [17 April, 2023], Labels:[DATE] 17 April, 2023 (tag: p)
Puretext: [Chat-GPT's Impact on Student and Professor Learning Experience.], Labels:[PERSON] Chat-GPT's (tag: a)
Puretext: [29 April, 2023], Labels:[DATE] 29 April, 2023 (tag: p)
Puretext: [Want to live in the EU? New rules could make it easier to move between countries], Labels:[ORG] EU (tag: a)
Puretext: [Get in Touch] (tag: span)
Puretext: [CONTACT ME] (tag: h3)
Puretext: [I am based in Amsterdam, Netherlands, and I'm always looking to connect with new people. Whether you're interested in collaborating on a project or just want to say hello, I'd love to hear from you.], Labels:[GPE, GPE] Amsterdam, Netherlands (tag: p)
Puretext: [To contact me, please send an email to] (tag: p)
Email: [koki248vlog@gmail.com] (tag: a)
Puretext: [or reach out to me via social media DM. I make an effort to respond to all messages within 24 hours, so you can expect a prompt reply.], Labels:[TIME] 24 hours (tag: p)
Puretext: [Thank you for visiting my website, and I look forward to hearing from you soon!] (tag: p)
[Done parsing HTML file.]

Data for label 'PERSON': KOKI HIROSE, Koki Hirose, Koki, C++, Java, Deep Programming, CV, Java, Java, Chat-GPT, Actually Creating Jobs, Chat-GPT's
Data for label 'DATE': 2023, 2022-2023, 2020, 2017 - March 2019, 14 April, 2023, 15 April, 2023, 17 April, 2023, 29 April, 2023
Data for label 'LANGUAGE': English
Data for label 'Non-contextual': 日本語
Data for label 'ORG': Web Developer & Social, Degree & Major, International Foundation Year:, Science & Engineering, Bachelor of Human Development: Environmental Science, Views News, Latest News, EU
Data for label 'GPE': Japan, Amsterdam, Netherlands, Japan, LinkedIn, Amsterdam, Netherlands
Data for label 'PRODUCT': JavaScript, Tetris
Data for label 'Website': https://www.linkedin.com/in/koki-hirose-365a92240
Data for label 'Email': koki248vlog@gmail.com
Data for label 'TIME': 24 hours
[Done printing data by label.]
"""