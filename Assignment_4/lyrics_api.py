# Ian Annase
# 1/24/18

# musixmatch api base url
base_url = "https://api.musixmatch.com/ws/1.1/"

# your api key
api_key = "&apikey=ff86e427accfef27d5540dc78b2cbbbd"

# api methods
a1 = lyrics_matcher = "matcher.lyrics.get"
a2 = lyrics_track_matcher = "track.lyrics.get"
a3 = track_search = "track.search"
a4 = track_charts = "chart.tracks.get"
a5 = artist_charts = "chart.artists.get"
a6 = track_getter = "track.get"
api_methods = [a1,a2,a3,a4,a5,a6]

# format url
format_url = "?format=json&callback=callback"

# parameters
p1 = artist_search_parameter = "&q_artist="
p2 = track_search_parameter = "&q_track="
p3 = track_id_parameter = "&track_id="
p4 = artist_id_parameter = "&artist_id="
p5 = album_id_parameter = "&album_id="
p6 = has_lyrics_parameter = "&f_has_lyrics="
p7 = has_subtitle_parameter = "&f_has_subtitle="
p8 = page_parameter = "&page="
p9 = page_size_parameter = "&page_size="
p10 = word_in_lyrics_parameter = "&q_lyrics="
p11 = music_genre_parameter = "&f_music_genre_id="
p12 = music_language_parameter = "&f_lyrics_language="
p13 = artist_rating_parameter = "&s_artist_rating="
p14 = track_rating_parameter= "&s_track_rating="
p15 = quorum_factor_parameter = "&quorum_factor="
p16 = artists_id_filter_parameter = "&f_artist_id="
p17 = country_parameter = "&country="
p18 = release_date_parameter = "&s_release_date="
p19 = album_name_parameter = "&g_album_name="
paramaters = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19]

# arrays with paramaters for each method
x1 = lyrics_matcher_parameters = [p1,p2]
x2 = lyrics_track_matcher_parameters = [p3]
x3 = track_search_paramaters = [p1,p2,p10,p4,p11,p12,p12,p14,p15,p8,p9]
x4 = track_charts_paramaters = [p8,p9,p17,p6]
x5 = artist_charts_parameters = [p8,p9,p17]
x6 = track_getter_parameters = [p3]

# get the paramaters for the correct api method
def get_parameters(choice):
    if choice == a1:
        return x1
    if choice == a2:
        return x2
    if choice == a3:
        return x3
    if choice == a4:
        return x4
    if choice == a5:
        return x5
    if choice == a6:
        return x6
