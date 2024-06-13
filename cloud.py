#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:11:10 2023

@author: cywang"""

import re
import time
from datetime import datetime, timedelta
import requests
947
from bs4 import BeautifulSoup as soup
import AutoSummary as ausu
import jieba
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import ssl

#使用SSL module把證書驗證改成不需要驗證
ssl._create_default_https_context = ssl._create_unverified_context


#爬蟲網路機制
myheaders = {
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36',
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
  'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
  'Accept-Encoding': 'none',
  'Accept-Language': 'en-US,en;q=0.8',
  'Connection': 'keep-alive',
  'refere': 'https://example.com',
  'cookie': """your cookie value ( you can get that from your web page) """
}
"""
keyword = '中華職棒' #關鍵字

####ETtoday新聞雲 https://sports.ettoday.net/news-search.phtml?keywords=%E8%81%B7%E6%A3%92
url = "https://sports.ettoday.net/news-search.phtml?keywords="
url_endstr = ""
url = url + keyword + url_endstr #合併成可讀的url
html = requests.get(url, headers=myheaders, timeout=10)
html.encoding = 'utf-8'


# 確認是否下載成功
urls = []
if html.status_code == requests.codes.ok:
    for pg in range(1, 10):
        page_url = "https://sports.ettoday.net/news-search.phtml?keywords="
        pageurl_endstr = "&idx=1&kind=10&page="
        page_url = page_url + keyword + pageurl_endstr + str(pg)
        pghtml = requests.get(page_url, headers=myheaders, timeout=10)
        pghtml.encoding = 'utf-8'
        if pghtml.status_code == requests.codes.ok:
            sp = soup(pghtml.text, 'html.parser')
            a_tags = sp.select('.part_pictxt_1 a')
            for a_tag in a_tags:  #取得新聞連結
                url = a_tag.get('href')
                if url.startswith('https://sports.ettoday.net/news/') and url not in urls:
                    urls.append(a_tag.get('href'))


#處裡獲得文本
stops = []
with open('stopWord_summar.txt','r', encoding='utf-8-sig') as f:  #停用詞庫
    for line in f.readlines():
        stops.append(line.strip())

current_date = datetime.now()
#previous_date = current_date - timedelta(days=3)
previous_date = current_date - timedelta(weeks=1)

i = 1
f = open('新聞摘要.txt', 'w', encoding='utf-8-sig', errors='ignore')
for url in urls:  #逐一取得新聞

    html = requests.get(url, headers=myheaders, timeout=10)
    html.encoding = 'utf-8'
    # 確認是否下載成功
    if html.status_code == requests.codes.ok:
        sp = soup(html.text, 'html.parser')
        data1 = sp.select('.story p')  #新聞內容
        myDateStr = sp.select('.date')
        for dstr in myDateStr:
            myDate = dstr.get_text().lstrip()

        dateNum = [int(s) for s in re.findall(r'-?\d+', myDate)]
        if (len(dateNum) > 4):
            newsDate = datetime(dateNum[0], dateNum[1], dateNum[2], dateNum[3], dateNum[4])
        elif (len(dateNum) > 2):
            newsDate = datetime(dateNum[0], dateNum[1], dateNum[2])
        else:
            newsDate = previous_date

        text = ''
        #只抓取過去"previous_date"的新聞
        if newsDate > previous_date:
            print('處理第 {} 則新聞'.format(i))
            print(sp.title.string)
            print('發布時間:', myDate)
            for d in data1:
                if d.text != '':  #有新聞內容
                    text += d.text

    # 檢查文章內容
    if text.strip():  # 如果文本不是空的
        #print('原始文章內容:', text)
        sentences, indexs = ausu.split_sentence(text)  #按標點分割句子
        tfidf = ausu.get_tfidf_matrix(sentences, stops)  #移除停用詞並轉換為矩陣
        word_weight = ausu.get_sentence_with_words_weight(tfidf)  #計算句子關鍵詞權重
        posi_weight = ausu.get_sentence_with_position_weight(sentences)  #計算位置權重
        scores = ausu.get_similarity_weight(tfidf)  #計算相似度權重
        sort_weight = ausu.ranking_base_on_weigth(word_weight, posi_weight, scores, feature_weight = [1,1,1])
        summar = ausu.get_summarization(indexs, sort_weight, topK_ratio = 0.3)  #取得摘要
        #print('摘要:', '\n')
        #print(summar)
        print('==========================================================')
        i += 1
        print(summar, file=f)
# #    else:
# #        print('文本為空，可能包含過多停用詞或者格式有問題。')
f.close()"""


#製圖文字雲
text = open('新聞摘要.txt', "r",encoding="utf-8-sig").read()  #讀文字資料
 
jieba.set_dictionary('dict.txt.big.txt')
jieba.load_userdict('user_dict_test.txt')
# 處理換行符，將其替換為空格
text = text.replace('\n', ' ')
with open('stopWord_summar.txt', 'r', encoding='utf-8-sig') as f:  #設定停用詞
    stops = f.read().split('\n')

terms = []  #儲存字詞
for t in jieba.cut(text, cut_all=False):  #拆解句子為字詞
    if t not in stops:  #不是停用詞
        terms.append(t)
diction = Counter(terms)

font = 'msjhbd.ttc'  #設定字型
#print(fon_path=font)
mask = np.array(Image.open("baseball.jpg"))  #設定文字雲形狀 
wordcloud = WordCloud(font_path=(font)) 
wordcloud = WordCloud(background_color="white",mask=mask,font_path=font)  #背景顏色預設黑色,改為白色 
wordcloud.generate_from_frequencies(frequencies=diction)  #產生文字雲

#產生圖片
plt.figure(figsize=(6,6))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

wordcloud.to_file("新聞總摘要_文字雲.png")  #存檔
