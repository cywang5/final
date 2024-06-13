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

# Load stopwords
stops = []
with open('stopWord_summar.txt', 'r', encoding='utf-8-sig') as f:  # Stopword list
    for line in f.readlines():
        stops.append(line.strip())

# Read the text and process it to get the summary
i = 1
with open('harrypotter.txt', 'r', encoding='utf-8-sig', errors='ignore') as text_file:
    text = text_file.read()

if text.strip():  # If the text is not empty
    sentences, indexs = ausu.split_sentence(text)  # Split the text into sentences
    tfidf = ausu.get_tfidf_matrix(sentences, stops)  # Remove stopwords and convert to matrix
    word_weight = ausu.get_sentence_with_words_weight(tfidf)  # Calculate keyword weights
    posi_weight = ausu.get_sentence_with_position_weight(sentences)  # Calculate position weights
    scores = ausu.get_similarity_weight(tfidf)  # Calculate similarity weights
    sort_weight = ausu.ranking_base_on_weigth(word_weight, posi_weight, scores, feature_weight=[1, 1, 1])
    summary = ausu.get_summarization(indexs, sort_weight, topK_ratio=0.3)  # Get the summary

    # Write the summary to a file
    with open('summary.txt', 'w', encoding='utf-8-sig') as f:
        print(summary, file=f)

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
mask = np.array(Image.open("hp.jpg"))  # Set the word cloud shape
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
wordcloud.to_file("hp.png")



