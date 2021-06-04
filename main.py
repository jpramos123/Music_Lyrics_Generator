import pandas as pd
from bs4 import BeautifulSoup
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from sklearn import preprocessing
import collections
from tqdm import tqdm
import numpy as np
import os
import re
import itertools
from nltk import ngrams


def remove_html(text):
    soup = BeautifulSoup(text, 'lxml')
    html_free = soup.text
    return html_free

def buildTransitionTable(row, graph, reverse_dict):
    phrase = row['VLyric']
    #context = row['type']

    for j in range(len(phrase)):
        # Valida se a palavra atual tem vizinho
        if j+1 >= len(phrase):
            pass
        else:
            
            # Se tem vizinho, incrementa na interseção
            graph[reverse_dict[phrase[j]]][reverse_dict[phrase[j+1]]] += 1
            graph[reverse_dict[phrase[j+1]]][reverse_dict[phrase[j]]] += 1


def buildEmissionTable(row, emissionTable, reverseDictGenres, reverseDictWords):
    phrase = row['VLyric']
    genre = row['Genre']

    for j in range(len(phrase)):
        emissionTable[reverseDictGenres[genre]][reverseDictWords[phrase[j]]] += 1


def remove_punctuation(text):
    clean_txt = "".join([c for c in text if c not in string.punctuation])
    return clean_txt

def makeGramCount(text, n, gram_dict):
    """ 
        This function return a dictionary with unique grams and the count of each
    """
    n_limit = n - 1
    loop_limit = len(text) - n_limit
    for i in range(loop_limit):
        current_seq = tuple(text[i:n+i])
        if current_seq in gram_dict.keys():
            gram_dict[current_seq] += 1
        else:
            gram_dict[current_seq] = 1

def generateNgramText(text, gram_arr, reverseDictWords, reverseDictGenre, genre, emissionTable, n_words):

    import numpy as np
    import math


    n = len(gram_arr)
    count = 0
    generatedText = []
    for count in tqdm(range(n_words), desc="Finding words", ncols=100):
        maxProb = -math.inf
        finalNextWord = ''
        for nextWord in reverseDictWords.keys():
            try:
                count_3 = gram_arr[2][(text[0], text[1], nextWord)]
                count_2 = gram_arr[1][(text[0], text[1])]
                count_1 = gram_arr[0][(text[0],)]
                emissionProb = emissionTable[reverseDictGenre[genre]][reverseDictWords[nextWord]]
            except KeyError:
                continue

            final_prob = np.log(count_3/count_2) + np.log(count_2/count_1) + np.log(count_1/total) + np.log(emissionProb)
            
            if final_prob > maxProb:
                maxProb = final_prob
                finalNextWord = nextWord
    
        generatedText.append(finalNextWord)  
        text[0] = text[1]
        text[1] = finalNextWord     

    return generatedText


if __name__ == "__main__":
    import sys

    in_text = input("Input 2 words: ")
    genre = input("Choose the genre (Rock, Pop or Hip Hop): ")
    n_words = int(input("How many words to generate? "))
    
    ## Get artists and lyrics data
    print("-- Getting data --")
    artists = pd.read_csv('artists-data.csv')
    lyrics = pd.read_csv('lyrics-data.csv')

    # Apply filter to data
    print("-- Filtering Text --")
    artists_filtered = artists[['Genre']].join(artists[['Link']].drop_duplicates())
    artists_filtered = artists_filtered.loc[~artists_filtered['Link'].isnull()]

    # Filter by English only and join Genre to lyrics
    en_lyrics = lyrics.loc[lyrics['Idiom'] == 'ENGLISH']
    en_lyrics = en_lyrics.join(artists_filtered.set_index('Link'), on='ALink')
    en_lyrics = en_lyrics.loc[en_lyrics['Genre'].isin(['Rock', 'Pop', 'Hip Hop'])]
    del lyrics


    #en_lyrics = pd.concat([en_lyrics[:200],en_lyrics[80000:80200],en_lyrics[-200:]])

    en_lyrics['Lyric'] = en_lyrics['Lyric'].str.lower()
    en_lyrics = en_lyrics[['Lyric','Genre']]

    print("-- Clearning Text --")
    en_lyrics['Lyric'] = en_lyrics['Lyric'].apply(lambda  text: remove_html(text))
    en_lyrics['Lyric'] = en_lyrics['Lyric'].apply(lambda  text: remove_punctuation(text))

    en_lyrics['VLyric'] = en_lyrics['Lyric'].apply(lambda  text: text.split())

    # Transforma as frases por linha em palavras por linha
    print("-- Creating Unique Words dict --")
    word_type = pd.DataFrame(en_lyrics['VLyric'].apply(pd.Series,1).stack())
    word_type = word_type.droplevel(1)
    word_type = word_type.merge(en_lyrics['Genre'].to_frame(),  left_index=True, right_index=True)

    word_type.rename(columns={0:"words"}, inplace=True)

    unique_words = word_type['words'].unique()
    reverse_dict_words = {unique_words[i]:i for i in range(len(unique_words))}
    unique_genres = en_lyrics['Genre'].unique()
    reverse_dict_genre = {unique_genres[i]: i for i in range(len(unique_genres))}

    print("-- Creating Emission Table by Genre --")
    emission_table = np.zeros((len(unique_genres), len(unique_words)), dtype=int)
    en_lyrics.apply(lambda row: buildEmissionTable(row, emission_table, reverse_dict_genre, reverse_dict_words), axis=1)
    emission_table_norm = preprocessing.normalize(emission_table, 'l1')

    threeGram = dict()
    biGram = dict()
    uniGram = dict()
    en_lyrics['VLyric'].apply(lambda row: makeGramCount(row, 3, threeGram))
    en_lyrics['VLyric'].apply(lambda row: makeGramCount(row, 2, biGram))
    en_lyrics['VLyric'].apply(lambda row: makeGramCount(row, 1, uniGram))


    # Count total of words in corpus
    total = 0
    for w in uniGram.keys():
        total = total + uniGram[w]


    print(generateNgramText(in_text.split(), 
                    [uniGram, biGram, threeGram], 
                    reverse_dict_words, 
                    reverse_dict_genre, 
                    genre, 
                    emission_table_norm,
                    n_words))