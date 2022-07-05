#!/usr/bin/env python
# coding: utf-8

'''
Spotify Reommendation Engine -- Spotify API Data Importor
Author: Yumo Bai
Email: baiym104@gmail.com
Date: July 2, 2022

This is the script to import the data used for Song Mood classification and the user's top items.
The data are imported with Spotify Web API using the spotipy library.

This process cannot be easily reproduced due to the complexity of Spotify's authorization framework,
but the produced datasets are stored under the data folder as mood_tracks.csv and user_top_tracks.csv.

You can find Spotify Web API here: https://developer.spotify.com/documentation/web-api/reference/#/
'''

from socket import SO_LINGER
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# spotipy object to extract data from Spotify Web API
# To set up the auth_manager it requires two variables: SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET to authenticate the connection
# I have the authentication info stored as envrionmental variables in my OS so this would not be naively reproduceable on every machine.
# To acquire authentication keys you would need to setup a Spotify devloper APP at https://developer.spotify.com/dashboard/
print('Connecting to Spotify...')
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

# The process of creating the labelled dataset is as followed:
# 1. Search for playlists with the name of the mood
# 2. Extract tracks from found playlists and label them with the searched mood's name
# 3. Combine the tracks and get their audio features using Spotify Web API
# 4. Optional: Find the genres of the associated artist as the genre of the Track (This is not implemented for the labelled
# dataset since the mood classification was only trained on part of the available track features but it could 
# be helpful if we were to expand the scope of features asscociated with the song mood classification)

moods = ['Happy', 'Sad', 'Energetic', 'Calm']

def generate_mood_lists(moods):
    mood_lists = {}
    for mood in moods:
        mood_lst = []
        
        for item in sp.search(mood, type='playlist')['playlists']['items']:
            mood_lst.append(item['id'])
        
        mood_lists[mood] = mood_lst
    return mood_lists

print('Generating mood lists...')
mood_lists = generate_mood_lists(moods)

def get_tracks_from_playlist(playlist, mood, time=None):
    """Generates a numpy array of the tracks from a playlist

    Args:
        items (Dict): Json formatted items
        mood (str): The name of the mood to be labelled for the tracks in that playlist

    Returns:
        tracks (list): The track information
    """
    
    tracks = []

    items = playlist['items']
    
    for item in items:
        # Skip tracks that have information missing
        try:

            # If time is None then we are searching for song mood track features
            if time == None:
                track = item['track']
                track_name = track['name']
                track_id = track['id']

                # only the first artist is considered the actual artist
                artist = track['artists'][0]
                artist_name = artist['name']
                artist_id = artist['id']

                album_name = track['album']['name']
                album_id = track['album']['id']

                popularity = track['popularity']

                info = [track_name, track_id, artist_name, artist_id, album_name, album_id, popularity, mood]
                tracks.append(info)

            # if time is not none we are searching for user top items
            else:
                track_name = item['name']
                track_id = item['id']

                # only the first artist is considered the actual artist
                artist = item['artists'][0]
                artist_name = artist['name']
                artist_id = artist['id']

                album_name = item['album']['name']
                album_id = item['album']['id']

                popularity = item['popularity']

                info = [track_name, track_id, artist_name, artist_id, album_name, album_id, popularity, mood, time]
                tracks.append(info)
        except TypeError:
            continue
    
    tracks = np.array(tracks)
    
    return tracks


def generate_track_df(mood_lists):
    """Generate the dataframe of all the songs from the found playlists

    Args:
        mood_lists (dict): Each mood and playlists associated with it

    Returns:
        track_df (DataFrame): labelled dataset for song mood classification
    """
    track_df = pd.DataFrame([])
    
    for mood, playlists in mood_lists.items():
        for playlist in playlists:
            items = sp.playlist_tracks(playlist, fields=['items'])
            tracks = get_tracks_from_playlist(items, mood)
            df = pd.DataFrame(tracks, columns=[
                'track_name', 'track_id', 'artist_name', 'artist_id',
                'album_name', 'album_id', 'popularity', 'mood'])
            
            track_df = pd.concat([track_df, df])
    
    track_df.reset_index(inplace=True)
    return track_df

print('Building track dataframe...')
track_df = generate_track_df(mood_lists)

# drop songs that appear in multiple playlists
track_df = track_df.drop_duplicates()


def generate_features(df, track_id):
    """Generate audio features from the track df.

    Args:
        df (DataFrame): track dataframe which stores the identifiers of tracks
        track_id (str): The name of the column whose values are to be used to search for track features.
                        In this project we used 'track_id'

    Returns:
        df (DataFrame): The input dataframe augmented with track features
    """
    features_list = [sp.audio_features(_id)[0] for _id in df[track_id]]
    
    # Some tracks does not generate features properly
    features_list = list(filter(None, features_list))
    
    features_df = pd.DataFrame(features_list)
    features_df.drop(columns=['uri','track_href', 'analysis_url'], inplace=True)
    features_df = features_df.rename(columns={'id':'track_id'})
    
    df = df.merge(features_df, on='track_id', how='left')
    
    if 'index' in df.columns:
        df = df.drop('index', axis=1)
    
    return df

print('Augmenting track features...')
track_df = generate_features(track_df, 'track_id')

def add_genres(df, artist_id):
    """Generate genres for tracks in a dataframe. The genres will be what the artist is associated with.

    Args:
        df (DataFrame): DataFrame where the tracks are stored
        artist_id (str): Name of the column whose values are to be used to find the artists.
                        In this project we used 'artist_id'
    """
    genres = []
    for _id in df[artist_id]:
        artist = sp.artist(_id)
        genres.append(artist['genres'])
    
    df['genres'] = genres

# As mentioned above, we don't need genres for song mood classification but we will need it later for generating user top tracks
# add_genres(track_df, 'artist_id')

track_df.to_csv('./data/song_mood_data.csv')
print('Song mood dataset generated successfully!')


### Get user top tracks

# Unfortunately Spotify only allows accessing the information of the current user. In this case me.
# So the testing for the recommendation part cannot be tested throughly.

# This dataset stores the user's top 10, 20 and 50 tracks for the time range of past 4 weeks, 6 month and several years 
# respectively. (Spotify defined these time ranges as short_term, medium_term and long_term)
# The idea behind picking 10, 20, and 50 for such time ranges is because the user's top tracks probably changes rapidly in the
# shorter terms, whereas they show a strong preference to a track if they have been consistently playing it to have it as a
# one of the top tracks across several years.
from spotipy.oauth2 import SpotifyOAuth
import os

print('Acquiring permissions for reading user top tracks...')
client_id = os.environ['SPOTIPY_CLIENT_ID']
client_secret = os.environ['SPOTIPY_CLIENT_SECRET']
username='Yumo-Bai'
scope = ['user-top-read','user-library-read']
redirect_uri = 'https://accounts.spotify.com/authorize/'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id, client_secret, redirect_uri, scope=scope))

time_rank_range = {'short_term':10, 'medium_term':20, 'long_term':50}

user_top_tracks = []

print('Retreiving user top tracks...')
for time, rank in time_rank_range.items():
    user_top_tracks.append(get_tracks_from_playlist(sp.current_user_top_tracks(limit=rank, 
                            time_range=time), None, time))

user_top_tracks = np.vstack(tuple(user_top_tracks))

user_top_tracks_df = pd.DataFrame(user_top_tracks, columns=[
                'track_name', 'track_id', 'artist_name', 'artist_id',
                'album_name', 'album_id', 'popularity', 'mood', 'time_range'])

print('Augmenting track features...')
user_top_tracks_df = generate_features(user_top_tracks_df, 'track_id')
user_top_tracks_df = user_top_tracks_df.drop_duplicates().reset_index().drop('index', axis=1)

print('Augmenting genres...')
add_genres(user_top_tracks_df, 'artist_id')

user_top_tracks_df.to_csv('./data/user_top_tracks.csv')
print('User top tracks data imported successfully!')