# import the Flask class from the flask module
from flask import Flask, render_template, request
from lyrics_api import *
import requests
import pandas as pd
from bs4 import BeautifulSoup
from constants import (
    TOKEN
)
import re
from PyLyrics import *
import os

from tensorflow import keras
from load_vectorize_data import *

defaults = {
    'request': {
        'token': TOKEN,
        'base_url': 'https://api.genius.com'
    }
}

def get_lyrics_GeniusAPI(artist_name,song_title):
        try:
            response = request_song_info(song_title, artist_name)
            json = response.json()
            remote_song_info = None
    
            for hit in json['response']['hits']:
                if artist_name.lower() in hit['result']['primary_artist']['name'].lower():
                    remote_song_info = hit
                    break
    
        # Extract lyrics from URL if song was found
            if remote_song_info:
                song_url = remote_song_info['result']['url']
                lyrics = scrap_song_url(song_url) 
            
            lyrics = re.sub(r'\[.*?\]', '', lyrics)     
            return lyrics
        
        except:
            return 'exception'
        
def get_lyrics_LyricsWikiaAPI(artist_name,song_title):
    try:
        lyrics = PyLyrics.getLyrics(artist_name,song_title)
        lyrics = lyrics.strip().replace('\n', ' ').replace('\r', ' ')
        return lyrics
        
    except:
        return 'exception'
    
def request_song_info(song_title, artist_name):
    base_url = defaults['request']['base_url']
    headers = {'Authorization': 'Bearer ' + defaults['request']['token']}
    search_url = base_url + '/search'
    data = {'q': song_title + ' ' + artist_name}
    response = requests.get(search_url, data=data, headers=headers)

    return response

def scrap_song_url(url):
    page = requests.get(url)
    html = BeautifulSoup(page.text, 'html.parser')
    [h.extract() for h in html('script')]
    lyrics = html.find('div', class_='lyrics').get_text().strip().replace('\n', ' ').replace('\r', ' ')

    return lyrics


def call_model(df, dict_to_append, data_tohtml):
    df_new = pd.DataFrame()
    cntHappy = 0
    cntSad = 0
    mood = ''
       
    x_test_ngram_vec = ngram_vectorize_pred(df)
    
    model = keras.models.load_model('music_classification_model.h5') 
    y_pred = model.predict(x_test_ngram_vec)   
    print(y_pred)
    
    index = 0
    for pred in y_pred:
       if(pred > 0.5):
           dict_mood = {"track_mood": 'happy'}
           cntHappy = cntHappy + 1
           dict_to_append = data_tohtml[index]
           dict_to_append.update(dict_mood)
           df_new = df_new.append(pd.DataFrame(dict_to_append, index=[0]), ignore_index=True)
       else:
           dict_mood = {"track_mood": 'sad'}
           cntSad = cntSad + 1
           dict_to_append = data_tohtml[index]
           dict_to_append.update(dict_mood)
           df_new = df_new.append(pd.DataFrame(dict_to_append, index=[0]), ignore_index=True)
       index = index  + 1
       
    if(cntSad > cntHappy):
        mood = 'Sad'
    else:
        mood = 'Happy'
    
            
    return data_tohtml, mood

def call_model_for_custom(lyrics):
    df_new = pd.DataFrame()
    mood = ''
    probability = 0
    dict_customlyrics = {"lyrics": lyrics}
    df_new = df_new.append(pd.DataFrame(dict_customlyrics, index=[0]), ignore_index=True)
            
    x_test_ngram_vec = ngram_vectorize_pred(df_new)
    
    model = keras.models.load_model('music_classification_model.h5') 
    y_pred = model.predict(x_test_ngram_vec) 
    
    
    for pred in y_pred:
       if(pred > 0.5):
           mood = 'Happy'
           probability = pred * 100
       else:
            mood = 'Sad'
            probability = (pred + 0.5) * 100
        
    return probability, mood
    
# create the application object
app = Flask(__name__)
app._static_folder = os.path.abspath("templates/static/")
 
@app.route('/')
def home():
    return render_template('index.html')

# use decorators to link the function to a url
@app.route('/results', methods=['GET', 'POST'])
def index():
    from flask import request
    df = pd.DataFrame()
    lyrics = ''
    data_tohtml = []
    dict_to_append = {}
    country = request.args.get('country') #if key doesn't exist, returns None
    kvalue = request.args.get('kvalue')
    cn = ''
    
    if(country == '' or kvalue == '' or int(kvalue) < 1 or not(kvalue.isdigit())):
       return render_template('index.html')   
   
    
    if(country == 'australia'):
        cn = 'au'
        country = 'Australia'
    elif(country == 'canada'):
        cn = 'ca'
        country = 'Canada'
    else:
        cn = 'gb'
        country = 'United Kingdom'
        
    # start building the api call
    api_call = base_url + track_charts + format_url
    api_call = api_call + page_parameter + '1'
    api_call = api_call + page_size_parameter + kvalue
    api_call = api_call + country_parameter + cn
    api_call = api_call + has_lyrics_parameter + '1'
    api_call = api_call + '&format=' + 'json'
    api_call = api_call + api_key
    toptracks_initial = requests.get(api_call).json()
    keys_filter = ['track_name', 'album_name', 'artist_name','track_share_url']# 'track_id', 'commontrack_id'
    
    #toptracks_json = json.loads(data)
    toptracks_initial = toptracks_initial['message']
    toptracks_initial = toptracks_initial['body']
    toptracks_initial = toptracks_initial['track_list']
    for tracks in toptracks_initial:
        track = tracks['track']
        dict_to_append = { your_key: track[your_key] for your_key in keys_filter }
        artist_name_fromdict = dict_to_append['artist_name'].split('feat', 1)[0].strip() 
        artist_name_fromdict = artist_name_fromdict.split(',', 1)[0].strip()
        artist_name_fromdict = artist_name_fromdict.split('(', 1)[0].strip()
        
        track_name_fromdict = dict_to_append['track_name'].split('-', 1)[0].strip()
        track_name_fromdict = track_name_fromdict.split(',', 1)[0].strip()
        track_name_fromdict = track_name_fromdict.split('(', 1)[0].strip()
        
        song_title, artist_name = track_name_fromdict, artist_name_fromdict
        
        lyrics = get_lyrics_LyricsWikiaAPI(artist_name,song_title)
        if(lyrics == 'exception' or lyrics == ''):
            lyrics = get_lyrics_GeniusAPI(artist_name,song_title)
            lyrics = re.sub('^\[*\]$', '', lyrics) 
            
        if(lyrics != 'exception' or lyrics != ''):
            dict_lyrics = {"lyrics": lyrics}
            dict_to_append.update(dict_lyrics)
            data_tohtml.append(dict_to_append)
            df = df.append(pd.DataFrame(dict_to_append, index=[0]), ignore_index=True)    
            
        
    
    data_tohtml, mood = call_model(df, dict_to_append, data_tohtml)
    
    return render_template("results.html", data = data_tohtml, country = country, mood = mood)


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/custom')
def custom():
    return render_template('custom.html')

@app.route('/customresults', methods=['GET', 'POST'])
def customresults():
    lyrics = ''   
    if request.method == 'POST':
        lyrics = request.form['lyrics']
    
    probability, mood = call_model_for_custom(lyrics)
    return render_template('customresults.html', probability = str(probability), mood = mood)

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)
