#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import os
from recflix import utils
from books.models import Book
from movies.models import Movie
from tvshows.models import *

data_path=os.path.abspath('datasets/tvshow_data.csv')
rating_path=os.path.abspath('datasets/tvshow_ratings.csv')

def get_show_details(title):
    qSet=Show.objects.filter(show_title=title)
    qSet=qSet[0]
    #print(qSet)
    ans={}
    ans['show_title']=qSet.show_title
    ans['show_id']=qSet.show_id
    ans['show_plot']=qSet.show_plot
    ans['show_genre']=qSet.show_genre
    ans['show_link']=qSet.show_link
    ans['show_rating']=qSet.show_rating
    return ans

def popular_shows():
    data = pd.read_csv(data_path)
    data = data.sort_values('show_rating', ascending=False)
    ans = []
    i = 0
    for index, row in data.iterrows():
        i += 1
        if (i >= 7):
            break
        ans.append(row['show_name'])
    final=[]
    for i in ans:
        final.append(get_show_details(i))
    return final


def top_charts(genre):
    data = pd.read_csv(data_path)
    data = data.sort_values('show_rating', ascending=False)
    ans = []
    for index, row in data.iterrows():
        if (genre in row['show_genre']):
            ans.append(row['show_name'])
    final=[]
    for i in ans[:6]:
        final.append(get_show_details(i))
    return final


def clean_genre(s):
    return s.replace(' ', '').split(',')


def similar_shows(title):
    data = pd.read_csv(data_path)
    data = data.drop('show_rating', axis=1)
    data = data.drop('show_link', axis=1)
    data = data.drop('show_id', axis=1)
    data['show_genre'] = data['show_genre'].map(lambda x: clean_genre(x))
    data['key_words'] = ""

    key_words_list = []
    for index, row in data.iterrows():
        plot = row['show_plot']
        r = Rake()
        r.extract_keywords_from_text(plot)
        key_words_dict_scores = r.get_word_degrees()
        key_words_list.append(list(key_words_dict_scores.keys()))
    data['key_words'] = key_words_list

    data.drop(columns=['show_plot'], inplace=True)
    data.set_index('show_name', inplace=True)
    data['bag_of_words'] = ''
    columns = data.columns
    for index, row in data.iterrows():
        words = ''
        for col in columns:
            words = words + ' '.join(row[col]) + ' '
        data.at[index, 'bag_of_words'] = words

    data.drop(columns=[col for col in data.columns if col != 'bag_of_words'], inplace=True)
    count = TfidfVectorizer()
    count_matrix = count.fit_transform(data['bag_of_words'])
    indices = pd.Series(data.index)
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top_10_indexes = list(score_series.iloc[1:6].index)
    ans=[]
    for i in top_10_indexes:
        ans.append(data.iloc[i].name)
    final=[]
    for i in ans:
        final.append(get_show_details(i))
    return final


def personalized_shows(username):
    try:
        from collections import Counter
        rated = Show_Rating.objects.filter(username=username)

        if len(rated) < 3:
            return 'not_enough'

        genre_weights = Counter()
        genre_show_count = Counter()
        already_rated = []

        for r in rated:
            already_rated.append(r.show_id)
            show = Show.objects.filter(show_id=r.show_id).first()
            if show:
                genres = show.show_genre.split(',')
                rating = float(r.rating)
                for g in genres:
                    g = g.strip()
                    genre_show_count[g] += 1
                    if rating >= 5:
                        genre_weights[g] += 2
                    elif rating >= 4:
                        genre_weights[g] += 1
                    elif rating <= 2:
                        genre_weights[g] -= 1

        qualified_genres = {g: w for g, w in genre_weights.items()
                           if genre_show_count[g] >= 2 and w > 0}

        if not qualified_genres:
            all_shows = Show.objects.all()
            candidates = [s for s in all_shows if s.show_id not in already_rated]
            candidates.sort(key=lambda x: float(x.show_rating) if x.show_rating else 0, reverse=True)
            final = []
            for show in candidates[:12]:
                details = get_show_details(show.show_title)
                if details:
                    final.append(details)
            return final

        sorted_genres = sorted(qualified_genres.items(), key=lambda x: x[1], reverse=True)
        primary_genre = sorted_genres[0][0]
        secondary_genre = sorted_genres[1][0] if len(sorted_genres) > 1 else None

        all_shows = Show.objects.all()
        candidates = []
        for show in all_shows:
            if show.show_id not in already_rated:
                if primary_genre in show.show_genre:
                    candidates.append(show)

        if len(candidates) < 5 and secondary_genre:
            for show in all_shows:
                if show.show_id not in already_rated and show not in candidates:
                    if secondary_genre in show.show_genre:
                        candidates.append(show)

        candidates.sort(key=lambda x: float(x.show_rating) if x.show_rating else 0, reverse=True)

        final = []
        for show in candidates[:12]:
            details = get_show_details(show.show_title)
            if details:
                final.append(details)
        return final
    except:
        return []

def rate_show(username,show_id,rating):
    qSet=Show_Rating.objects.filter(username=username,show_id=show_id)
    if(len(qSet)==0):
        old = open(rating_path,'a')
        old.write(str(username) + "," + str(show_id) + "," + str(rating) + "\n")
        old.close()
        obj = Show_Rating(username=str(username), show_id=str(show_id), rating=str(rating))
        obj.save()
    else:
        #to update rating csv file
        qSet[0].rating=rating
        qSet[0].save()
        with open(rating_path, 'r') as f:
            data = f.readlines()
            f.close()
        for i in range(len(data)):
            if ((username + ',' + show_id) in data[i]):
                data[i] = username + ',' + show_id + ',' + rating+'\n'
        with open(rating_path, 'w') as file:
            file.writelines(data)
            file.close()

def get_similar_content(show_id):
    items=utils.similar_items(show_id)
    book_ids=[]
    movie_ids=[]
    for i in items:
        if i[0]=='m':
            if(len(movie_ids)!=5):
                movie_ids.append(i)
        elif i[0]=='b':
            if(len(book_ids)!=5):
                book_ids.append(i)
        if(len(book_ids)==5 and len(movie_ids)==5):
            break

    similar_books=[]
    similar_movies=[]

    for i in book_ids:
        qSet=Book.objects.filter(book_id=i)[0]
        ans = {}
        ans['book_title'] = qSet.book_title
        ans['book_id'] = qSet.book_id
        ans['book_plot'] = qSet.book_plot
        ans['book_genre'] = qSet.book_genre
        ans['book_link'] = qSet.book_link
        ans['book_rating'] = qSet.book_rating
        similar_books.append(ans)

    for i in movie_ids:
        qSet = Movie.objects.filter(movie_id=i)[0]
        ans = {}
        ans['movie_title'] = qSet.movie_title
        ans['movie_id'] = qSet.movie_id
        ans['movie_plot'] = qSet.movie_plot
        ans['movie_genre'] = qSet.movie_genre
        ans['movie_link'] = qSet.movie_link
        ans['imdb_rating'] = qSet.imdb_rating
        similar_movies.append(ans)

    return similar_movies,similar_books
