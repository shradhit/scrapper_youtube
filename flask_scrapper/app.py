import os
import glob

import string
import random

import youtube_dl

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

import webvtt
import numpy as np
import pandas as pd
import validators


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from flask import Flask, request, render_template, jsonify
from flask import Flask, render_template, Response, request, redirect, url_for


app = Flask(__name__)


@app.route('/')
def index():
    return '<h1> Youtube Sentimental Analysis!</h1>'


@app.route('/form_data')
def my_form():
    return render_template('template.html')


@ app.route("/form_data", methods=["GET", "POST"])
def form_data():

    if request.method == "POST":
        link = request.form['link']
        if validators.url(link):
            link_received = link
        else:
            return render_template("template3.html")

    file_image = randomString(8)

    source = os.getcwd()
    ydl_opts = {'writesubtitles': True,
                'writeautomaticsub': True,
                'writeinfojson': True,
                'format': 'bestaudio/best',
                'keepvideo': False,
                'postprocessors': [{'key': 'FFmpegExtractAudio',
                                    'preferredcodec': 'wav',
                                    'preferredquality': '192'}],
                'postprocessor_args': ['-ar', '16000']}

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        meta = ydl.extract_info(link_received, download=True)

    print('DONE')

    keys = ['uploader', 'uploader_url', 'upload_date', 'creator', 'title', 'description', 'categories',
            'duration', 'view_count', 'like_count', 'dislike_count', 'average_rating', 'start_time', 'end_time',
            'release_date', 'release_year']



    filtered_d = dict((k, meta[k]) for k in keys if k in meta)
    df = pd.DataFrame.from_dict(filtered_d, orient='index').T
    df.index = df['title']
    title = meta['title']
    files = os.listdir(source)

    for f in files:
        if f.endswith('vtt'):
            file_envtt = './down/' + file_image + '.en.vtt'
            os.rename(f, file_envtt)

        elif f.endswith('json'):
            file_json = './down/' + file_image + '.json'
            os.rename(f, file_json)

        elif f.endswith('wav'):
            file_wav = './down/' + file_image + '.wav'
            os.rename(f, file_wav)

        else:
            continue




    sub_titles = glob.glob('./down/' + file_image + '.en.vtt')
    print(sub_titles)
    if len(sub_titles) != 0:
        vtt = webvtt.read(sub_titles[0])

        start_list = list()
        end_list = list()
        # Storing all the lines as part of the lines list
        lines = []
        print('DONE 2')
        for x in range(len(vtt)):
            start_list.append(vtt[x].start)
            end_list.append(vtt[x].end)
        print('DONE 3')

        for line in vtt:
            lines.append(line.text.strip().splitlines())

        lines = [' '.join(item) for item in lines]
        print('DONE 4')

        final_df = pd.DataFrame({'Start_time': start_list, 'End_time': end_list, 'Statement': lines})

        sid_obj = SentimentIntensityAnalyzer()
        sentiment_scores_vader = [sid_obj.polarity_scores(article) for article in final_df.Statement]
        print('DONE 5')

        sentiment_category_positive = []
        sentiment_category_neutral = []
        sentiment_category_negative = []
        sentiment_category_compound = []

        for sentiments in sentiment_scores_vader:
            sentiment_category_positive.append(sentiments['pos'])
            sentiment_category_neutral.append(sentiments['neu'])
            sentiment_category_negative.append(sentiments['neg'])
            sentiment_category_compound.append(sentiments['compound'])

        sentiment_df = pd.DataFrame([[article for article in final_df.Statement],
                                     sentiment_category_positive,
                                     sentiment_category_neutral,
                                     sentiment_category_negative,
                                     sentiment_category_compound]).T
        sentiment_df['Start_time'] = start_list
        sentiment_df['End_time'] = end_list

        sentiment_df.columns = ['Statement', 'positive_polarity', 'neutral_polarity', 'negative_polarity',
                                'overall_polarity', 'Start_time', 'End_time']

        sentiment_df['Sentiment'] = ['Positive' if w>0 else 'Negative' if w<0 else 'Neutral' for w in sentiment_df['overall_polarity']]
        #abcd = sentiment_df.to_json()
        print('DONE 6')

        ######## Sentiment Outputs #############

        # Overall Sentiment of the video
        overall_sentiment = round(sentiment_df[sentiment_df['overall_polarity']!=0].overall_polarity.mean(),2)

        # Total sentences in the video
        total_sentences = sentiment_df.shape[0]
        positive_sentences_ = round(sentiment_df[sentiment_df['overall_polarity']>0].shape[0]/total_sentences, 2)
        positive_sentences = round(positive_sentences_ * 100, 2)
        neutral_sentences = round(sentiment_df[sentiment_df['overall_polarity'] == 0].shape[0]/total_sentences, 2)*100
        negative_sentences = round(sentiment_df[sentiment_df['overall_polarity'] < 0].shape[0]/total_sentences, 2)*100


        duration = round(df['duration'][0]/60,2)
        view_count = df['view_count'][0]
        like_count = df['like_count'][0]
        dislike_count = df['dislike_count'][0]
        average_rating = round(df['average_rating'][0],2)

        # Distribution
        distribution_df = sentiment_df
        list_color = ["r","g",'y']
        index = 0
        fig,ax = plt.subplots(figsize=(10,5))
        for group in distribution_df.Sentiment.unique():
            sns_distplot = sns.distplot(distribution_df.loc[distribution_df.Sentiment==group,'overall_polarity'],kde=False,ax=ax,label=group, color=list_color[index])
            index = index + 1
        plt.legend()
        plt.xlabel('Polarity Score')
        plt.ylabel('Number of sentences')
        plt.title('Distribution of the polarity score of the entire video', weight='bold')

        fig1 = sns_distplot.get_figure()
        img_save1 = file_image + "one.png"

        fig1.savefig("static/" + img_save1)

        print('DONE 6.5')

        heatmap_polarity = sentiment_df['overall_polarity'].astype('float').values
        heatmap_polarity = heatmap_polarity.reshape(heatmap_polarity.shape[0], 1)

        plt.figure(figsize=(16,5))
        sns_plot = sns.heatmap(data=heatmap_polarity.T, robust=True, cmap='RdYlGn',yticklabels=False, xticklabels=5, cbar=True, cbar_kws={"orientation": "horizontal"})
        plt.title('Heatmap reflecting the change in polarity of the speech', weight='bold')
        plt.xlabel('Sentences over time')

        fig2 = sns_plot.get_figure()


        img_save2 = file_image +"two.png"

        fig2.savefig("static/" + img_save2)

        print('DONE 7')

        #########

        comment_words = ' '
        stopwords = set(STOPWORDS)

        # iterate through the corpus
        for val in sentiment_df.Statement:

            # typecaste each val to string
            val = str(val)

            # split the value
            tokens = val.split()

            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()

            for words in tokens:
                comment_words = comment_words + words + ' '

        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=stopwords,
                              min_font_size=10).generate(comment_words)


        # Plot the WordCloud image
        #plt.title('Word Cloud', weight='bold')

        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.title("Word Cloud of Frequent Words", fontweight='bold')
        img_save3 = file_image + "three.png"

        plt.savefig("static/" + img_save3)

        #return render_template("template.html", graph_one=img_save1, graph_two=img_save2)
        print(img_save1)
        #delete_files(file_name)
        return render_template("template2.html", graph_one=img_save1, graph_two=img_save2,graph_three=img_save3,
                               v1=overall_sentiment,v2=total_sentences,v3=positive_sentences,v4=neutral_sentences,v5=negative_sentences,
                               v6=duration, v7=view_count,v8=like_count,v9=dislike_count,v10=average_rating, v11 = title)
    return render_template("template3.html")


def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))



def delete_files(random_name):
    dir_name = './down'
    all_files = os.listdir(dir_name)
    for item in all_files:
        if item.statswith(random_name):
            os.remove(os.path.join(dir_name, item))
            print("Deleted {}" .format(item))
    
