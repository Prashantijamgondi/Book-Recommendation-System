# import the libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

st.header("Book Recommender System")

st.markdown('''
  ##### The site using colaborative filtering suggrests books from our catalog.
  ##### We recommend top 50 books for every one as well
''')

# import our models :

popular=pickle.load(open('Models/popular.pkl', 'rb'))
books=pickle.load(open('Models/books.pkl', 'rb'))
pt=pickle.load(open('Models/pt.pkl', 'rb'))
similarity_scores=pickle.load(open('Models/similarity_scores.pkl', 'rb'))

# Top 50 books :

st.sidebar.title("Top 50 Books")

if st.sidebar.button('SHOW'):
    cols_per_row=5
    num_rows=10
    for row in range(num_rows):
        cols=st.columns(cols_per_row)
        for col in range(cols_per_row):
            book_idx=row* cols_per_row + col
            if book_idx < len(popular):
                with cols[col]:
                    st.image(popular.iloc[book_idx]['Image-URL-M']) # Diplays the image 
                    st.text(popular.iloc[book_idx]['Book-Title']) # Displays the book title
                    st.text(popular.iloc[book_idx]['Book-Author']) # Displays the author