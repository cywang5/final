import urllib.request as req
from urllib import parse as parse
from bs4 import BeautifulSoup as soup
import AutoSummary as ausu
import jieba
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import time
from datetime import datetime
import re

# Load stopwords
stops = []
with open('stopWord_summar.txt', 'r', encoding='utf-8-sig') as f:  # Stopword list
    for line in f.readlines():
        stops.append(line.strip())

# Read the text and process it to get the summary
i = 1
with open('harrypotter2.txt', 'r', encoding='utf-8-sig', errors='ignore') as text_file:
    text = text_file.read()

if text.strip():  # If the text is not empty
    sentences, indexs = ausu.split_sentence(text)  # Split the text into sentences
    tfidf = ausu.get_tfidf_matrix(sentences, stops)  # Remove stopwords and convert to matrix
    word_weight = ausu.get_sentence_with_words_weight(tfidf)  # Calculate keyword weights
    posi_weight = ausu.get_sentence_with_position_weight(sentences)  # Calculate position weights
    scores = ausu.get_similarity_weight(tfidf)  # Calculate similarity weights
    sort_weight = ausu.ranking_base_on_weigth(word_weight, posi_weight, scores, feature_weight=[1, 1, 1])
    summary = ausu.get_summarization(indexs, sort_weight, topK_ratio=0.3)  # Get the summary



def extract_quoted_sentences(input_file, output_file, keyword):
    # Regular expression to find quoted sentences before or after the keyword
    pattern = re.compile(r'(".*?")\s*[^"]*\b' + re.escape(keyword) + r'\b|'+ re.escape(keyword) + r'\b\s*[^"]*(".*?")', re.IGNORECASE)

    with open(input_file, 'r') as file:
        text = file.read()

    matches = pattern.findall(text)
    
    # Flatten the list of tuples and filter out empty strings
    quoted_sentences = [quote for match in matches for quote in match if quote]

    with open(output_file, 'w') as file:
        for sentence in quoted_sentences:
            file.write(sentence + "\n")

# Example usage
input_file = 'harrypotter2.txt'  # Input text file
output_file = 'summary.txt'  # Output text file
keyword = 'Voldemort'  # Keyword to search for


# Generate the word cloud
text = open('summary.txt', "r", encoding="utf-8-sig").read()  # Read the summary text

jieba.set_dictionary('dict.txt.big.txt')
jieba.load_userdict('user_dict_test.txt')
text = text.replace('\n', ' ')  # Replace newlines with spaces

# Load stopwords again for jieba
with open('stopWord_summar.txt', 'r', encoding='utf-8-sig') as f:
    stops = f.read().split('\n')

# Tokenize the text and remove stopwords
terms = [t for t in jieba.cut(text, cut_all=False) if t not in stops]
diction = Counter(terms)

# Generate the word cloud
font = 'msjhbd.ttc'  # Set the font
mask = np.array(Image.open("lv.jpg"))  # Set the word cloud shape
wordcloud = WordCloud(
    font_path=font,
    background_color="white",
    mask=mask
).generate_from_frequencies(frequencies=diction)

# Display the word cloud
plt.figure(figsize=(8, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Save the word cloud image
wordcloud.to_file("lv.png")



