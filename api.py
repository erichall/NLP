import logging
from logging.config import fileConfig
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
import re
fileConfig('logging.config.ini')
log = logging.getLogger('api') 

class Api(object):
    def __init__(self):
        self.url_base = 'http://lyrics.wikia.com'
        self.url_search = '/wiki/Special:Search?'

        print("new API object created")
        text = self.get_all_lyrics_from_artis("eminem")
        f = open('data/eminem','w')
        print(text)
        f.write(text[0])
        f.close()

    # find the top most result and fetches that artists songs
    def get_all_tracks_by_artist(self, artist):
        artist_url = self._search(artist)
        if not artist_url: # nothing found
            return []
        artist_page = self.get('',artist_url[0])

        soup = BeautifulSoup(artist_page, 'html.parser')
        tracks = soup.find_all('ol') 
        links = []
        for album in tracks:
            for track in album:
                link = track.find('a')
                if link:
                    links.append(link.get('href'))
        print(len(links))
        return links
        
    # return all lyrics from artist as text
    def get_all_lyrics_from_artis(self, artist):
        tracks = self.get_all_tracks_by_artist(artist)
        log.debug("Found: " + str(len(tracks)) +" tracks ")
        lyrics = []
        for track_url in tracks:
            lyric = self.get_lyric_from_url(track_url)
            if lyric:
                lyrics.append(lyric)
        return lyrics

    # returns list with urls from result, [] if nothing found
    def search_artist(self, artist):
        return self._search(artist)

    # return lyric in text, if not found return ''
    # url format: http://lyrics.wikia.com/wiki/artist:track
    def get_lyric_from_url(self, url):
        lyric_page = self.get('', self.url_base + url)
        soup = BeautifulSoup(lyric_page, 'html.parser')
        lyric = soup.find_all('div', class_='lyricbox')
        if lyric:
            lyric = re.sub('<[^<]+?>', ' ', str(lyric[0])) # remove html tags
            return lyric
        return ''
    
    def get(self, values, url):
        log.debug(str(values) + ' ' + url)
        data = urllib.parse.urlencode(values)
        data = data.encode('utf-8')
        request = urllib.request.Request(url, data)
        response = urllib.request.urlopen(request)
        response_data = response.read()
        return response_data

    # private method to search
    # returns list with urls, [] if nothing found
    def _search(self, query):
        log.debug(str(query) + ' ' + self.url_base + self.url_search)
        page = self.get({'query': query }, self.url_base + self.url_search)
        soup = BeautifulSoup(page, 'html.parser')
        results = soup.find_all('ul', class_='Results')

        if not results: # nothing found
            return []
        res = []
        for r in results[0].find_all('a'):
            res.append(r.get('href'))
        return res 

