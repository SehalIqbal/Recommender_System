from django.shortcuts import render
from django.http import HttpResponse
from movies.models import Movie_Rating
from books.models import Book_Rating
from tvshows.models import Show_Rating
from movies import movies_utils
from books import books_utils
from tvshows import shows_utils
import pandas as pd

def rate(request):
    username = str(request.user)
    item_type = request.GET.get('item_type')
    item_id = request.GET.get('item_id')
    rating = request.GET.get('rating')

    if item_type == 'movie':
        movies_utils.rate_movie(username, item_id, rating)
    elif item_type == 'book':
        books_utils.rate_book(username, item_id, rating)
    elif item_type == 'show':
        shows_utils.rate_show(username, item_id, rating)

    return HttpResponse('ok')

def reset_ratings(request):
    username = str(request.user)

    # Delete from database
    Movie_Rating.objects.filter(username=username).delete()
    Book_Rating.objects.filter(username=username).delete()
    Show_Rating.objects.filter(username=username).delete()

    # Remove from CSV files
    for path, col in [
        ('datasets/movie_ratings.csv', 'username'),
        ('datasets/book_ratings.csv', 'username'),
        ('datasets/tvshow_ratings.csv', 'username'),
    ]:
        try:
            df = pd.read_csv(path)
            df = df[df['username'] != username]
            df.to_csv(path, index=False)
        except:
            pass

    return HttpResponse('ok')