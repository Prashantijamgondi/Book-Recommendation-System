# # import the libraries
# Enhanced Book Recommender System with Search, Details, Authentication & CSS

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import hashlib
import sqlite3
from datetime import datetime
import re

# Page Configuration
st.set_page_config(
    page_title="BookWise - Intelligent Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
def load_css():
    st.markdown("""
    <style>
    /* Main App Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: white;
        text-align: center;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Search Box Styling */
    .search-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Book Cards */
    .book-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
        height: 100%;
    }
    
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .book-title {
        font-weight: bold;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        color: #2c3e50;
        line-height: 1.2;
    }
    
    .book-author {
        color: #7f8c8d;
        font-size: 0.8rem;
        margin-top: 0.3rem;
        line-height: 1.2;
    }
    
    .book-rating {
        color: #f39c12;
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Authentication Forms */
    .auth-form {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        max-width: 400px;
        margin: 2rem auto;
    }
    
    /* Book Detail Page */
    .book-detail {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .detail-image {
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    
    /* Text Input Styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 0.5rem 1rem;
    }
    
    /* Hide Streamlit Menu */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Database setup for authentication
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, email TEXT, created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_favorites
                 (username TEXT, book_title TEXT, added_at TEXT)''')
    conn.commit()
    conn.close()

# Hash password
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Check password
def check_password(password, hashed):
    return hash_password(password) == hashed

# User registration
def register_user(username, password, email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?)", 
                 (username, hash_password(password), email, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

# User login
def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result and check_password(password, result[0]):
        return True
    return False

# Load models function with error handling
@st.cache_data
def load_models():
    try:
        popular = pickle.load(open('Models/popular.pkl', 'rb'))
        books = pickle.load(open('Models/books.pkl', 'rb'))
        pt = pickle.load(open('Models/pt.pkl', 'rb'))
        similarity_scores = pickle.load(open('Models/similarity_scores.pkl', 'rb'))
        return popular, books, pt, similarity_scores
    except FileNotFoundError:
        st.error("Model files not found. Please ensure all pickle files are in the 'Models' directory.")
        return None, None, None, None

# Search functionality
def search_books(query, books_df, limit=20):
    if query:
        # Search in book titles and authors
        mask = (books_df['Book-Title'].str.contains(query, case=False, na=False) | 
                books_df['Book-Author'].str.contains(query, case=False, na=False))
        results = books_df[mask].drop_duplicates('Book-Title').head(limit)
        return results
    return pd.DataFrame()

# Recommendation function
def recommend(book_name, pt, similarity_scores, books):
    try:
        index = np.where(pt.index == book_name)[0][0]
        similarity_items = sorted(list(enumerate(similarity_scores[index])), 
                                key=lambda x: x[1], reverse=True)[1:6]
        
        data = []
        for i in similarity_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            if not temp_df.empty:
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
                item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
                data.append(item)
        return data
    except:
        return []

# Get book details
def get_book_details(book_title, books):
    book_data = books[books['Book-Title'] == book_title].iloc[0]
    return book_data

# Display book card
def display_book_card(image_url, title, author, rating=None, book_index=None):
    with st.container():
        st.markdown('<div class="book-card">', unsafe_allow_html=True)
        
        # Handle missing images
        try:
            st.image(image_url, width=150)
        except:
            st.image("https://via.placeholder.com/150x200?text=No+Image", width=150)
        
        st.markdown(f'<div class="book-title">{title[:50]}{"..." if len(title) > 50 else ""}</div>', 
                   unsafe_allow_html=True)
        st.markdown(f'<div class="book-author">{author[:30]}{"..." if len(author) > 30 else ""}</div>', 
                   unsafe_allow_html=True)
        
        if rating:
            st.markdown(f'<div class="book-rating">⭐ {rating}</div>', unsafe_allow_html=True)
        
        # Create a unique key for each button using title and a hash
        button_key = f"detail_{hash(title) % 10000}_{book_index if book_index else 0}"
        
        if st.button("View Details", key=button_key):
            st.session_state.selected_book_detail = title
            st.session_state.page = "book_detail"
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Authentication UI
def authentication_ui():
    st.markdown('<div class="auth-form">', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_btn"):
            if login_user(login_username, login_password):
                st.session_state.authenticated = True
                st.session_state.username = login_username
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Register")
        reg_username = st.text_input("Username", key="reg_username")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password")
        
        if st.button("Register", key="register_btn"):
            if reg_password != reg_confirm_password:
                st.error("Passwords don't match")
            elif len(reg_password) < 6:
                st.error("Password must be at least 6 characters")
            elif register_user(reg_username, reg_password, reg_email):
                st.success("Registration successful! Please login.")
            else:
                st.error("Username already exists")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Book detail page
def book_detail_page(book_title, books, pt, similarity_scores):
    try:
        book_data = get_book_details(book_title, books)
        
        st.markdown('<div class="book-detail">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            try:
                st.image(book_data['Image-URL-L'], width=300, caption=book_data['Book-Title'])
            except:
                st.image("https://via.placeholder.com/300x400?text=No+Image", width=300)
        
        with col2:
            st.title(book_data['Book-Title'])
            st.subheader(f"by {book_data['Book-Author']}")
            
            if 'Year-Of-Publication' in book_data:
                st.write(f"**Published:** {book_data['Year-Of-Publication']}")
            
            if 'Publisher' in book_data:
                st.write(f"**Publisher:** {book_data['Publisher']}")
            
            if 'Book-Rating' in book_data:
                rating = book_data['Book-Rating']
                if pd.notna(rating) and rating > 0:
                    st.write(f"**Average Rating:** {rating} ⭐")
            
            # Add to favorites button (if authenticated)
            if st.session_state.get('authenticated', False):
                if st.button("❤️ Add to Favorites"):
                    # Add favorite functionality here
                    st.success("Added to favorites!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show recommendations
        st.subheader("🔍 Similar Books You Might Like")
        recommendations = recommend(book_title, pt, similarity_scores, books)
        
        if recommendations:
            cols = st.columns(5)
            for col_idx, col in enumerate(cols):
                if col_idx < len(recommendations):
                    with col:
                        display_book_card(
                            recommendations[col_idx][2],
                            recommendations[col_idx][0],
                            recommendations[col_idx][1],
                            book_index=f"detail_rec_{col_idx}"
                        )
        
        if st.button("🔙 Back to Home"):
            st.session_state.page = "home"
            st.rerun()
            
    except Exception as e:
        st.error(f"Error loading book details: {str(e)}")
        if st.button("🔙 Back to Home"):
            st.session_state.page = "home"
            st.rerun()

# Main application
def main():
    # Initialize database
    init_db()
    
    # Load CSS
    load_css()
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Load models
    popular, books, pt, similarity_scores = load_models()
    
    if popular is None:
        return
    
    # Authentication check
    if not st.session_state.authenticated:
        st.markdown("""
        <div class="main-header">
            <h1>📚 BookWise</h1>
            <p>Discover Your Next Favorite Book</p>
        </div>
        """, unsafe_allow_html=True)
        
        authentication_ui()
        return
    
    # Navigation
    if st.session_state.page == "book_detail" and 'selected_book_detail' in st.session_state:
        book_detail_page(st.session_state.selected_book_detail, books, pt, similarity_scores)
        return
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>📚 BookWise</h1>
        <p>Welcome back, {}! Discover your next favorite book with AI-powered recommendations</p>
    </div>
    """.format(st.session_state.get('username', 'Reader')), unsafe_allow_html=True)
    
    # Logout button in sidebar
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
    
    # Search functionality
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.subheader("🔍 Search Books")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("", placeholder="Search by title or author...", key="search")
    with col2:
        search_button = st.button("Search", key="search_btn")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display search results
    if search_query or search_button:
        if search_query:
            search_results = search_books(search_query, books)
            
            if not search_results.empty:
                st.subheader(f"📖 Search Results ({len(search_results)} found)")
                
                # Display search results in grid
                cols_per_row = 5
                for i in range(0, len(search_results), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(search_results):
                            book = search_results.iloc[i + j]
                            with cols[j]:
                                display_book_card(
                                    book['Image-URL-M'],
                                    book['Book-Title'],
                                    book['Book-Author'],
                                    book_index=f"search_{i+j}"
                                )
            else:
                st.info("No books found matching your search.")
    
    # Sidebar content
    with st.sidebar:
        st.title("📊 Top 50 Popular Books")
        show_popular_btn = st.button('Show Top 50', key="show_popular")
        
        st.title("🎯 Get Recommendations")
        book_list = pt.index.values
        selected_book = st.selectbox("Select a book:", book_list, key="recommend_select")
        
        get_recommendations_btn = st.button('Get Recommendations', key="get_recommendations")
    
    # Handle button clicks and display results
    if show_popular_btn:
        st.subheader("🏆 Top 50 Most Popular Books")
        cols_per_row = 5
        num_rows = 10
        
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            for col in range(cols_per_row):
                book_idx = row * cols_per_row + col
                if book_idx < len(popular):
                    with cols[col]:
                        book = popular.iloc[book_idx]
                        rating = book.get('Avg-Rating', 'N/A')
                        display_book_card(
                            book['Image-URL-M'],
                            book['Book-Title'],
                            book['Book-Author'],
                            rating,
                            f"popular_{book_idx}"
                        )
    
    # Display recommendations
    if get_recommendations_btn:
        st.subheader(f"📚 Books Similar to '{selected_book}'")
        
        recommendations = recommend(selected_book, pt, similarity_scores, books)
        if recommendations:
            cols = st.columns(5)
            for col_idx, col in enumerate(cols):
                if col_idx < len(recommendations):
                    with col:
                        display_book_card(
                            recommendations[col_idx][2],
                            recommendations[col_idx][0],
                            recommendations[col_idx][1],
                            book_index=f"rec_{col_idx}"
                        )
        else:
            st.error("Sorry, couldn't find recommendations for this book.")

if __name__ == "__main__":
    main()
  
# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd

# st.set_page_config(layout="wide")

# st.header("Book Recommender System")

# st.markdown('''
#   ##### The site using colaborative filtering suggrests books from our catalog.
#   ##### We recommend top 50 books for every one as well
# ''')

# # import our models :

# popular=pickle.load(open('Models/popular.pkl', 'rb'))
# books=pickle.load(open('Models/books.pkl', 'rb'))
# pt=pickle.load(open('Models/pt.pkl', 'rb'))
# similarity_scores=pickle.load(open('Models/similarity_scores.pkl', 'rb'))

# # Top 50 books :

# st.sidebar.title("Top 50 Books")

# if st.sidebar.button('SHOW'):
#     cols_per_row=5
#     num_rows=10
#     for row in range(num_rows):
#         cols=st.columns(cols_per_row)
#         for col in range(cols_per_row):
#             book_idx=row* cols_per_row + col
#             if book_idx < len(popular):
#                 with cols[col]:
#                     st.image(popular.iloc[book_idx]['Image-URL-M']) # Diplays the image 
#                     st.text(popular.iloc[book_idx]['Book-Title']) # Displays the book title
#                     st.text(popular.iloc[book_idx]['Book-Author']) # Displays the author
                    
# def recommend(book_name):
#     index=np.where(pt.index==book_name)[0][0]
#     similarity_items=sorted(list(enumerate(similarity_scores[index])), key=lambda x:x[1], reverse=True)[1:6]
#     # lets create empy list and in that lies i want to populate with the book information
#     # Book author book-title image-url
#     # empty list
#     data=[]
#     for i in similarity_items:
#         item=[]
#         temp_df=books[books['Book-Title']==pt.index[i[0]]]
#         item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
#         item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
#         item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
#         data.append(item)
#     return data

# # This is giving the names list of books
# book_list=pt.index.values

# st.sidebar.title("Similar Book Suggestions")
# # Drop down to select the books
# selected_book=st.sidebar.selectbox("Select a book from the dropdown", book_list)

# if st.sidebar.button('RECOMMEND'):
#     book_recommend=recommend(selected_book)
#     cols=st.columns(5)
#     for col_idx in range(5):
#         with cols[col_idx]:
#             if col_idx<len(book_recommend):
#                 st.image(book_recommend[col_idx][2])
#                 st.text(book_recommend[col_idx][0])
#                 st.text(book_recommend[col_idx][1])
                

# # import data
# books=pd.read_csv('Data/Books.csv') # Books data
# ratings=pd.read_csv('Data/Ratings.csv') # Ratings data
# users=pd.read_csv('Data/Users.csv') # Users data'

# st.sidebar.title("Data Used")

# if st.sidebar.button("Show"):
#     st.subheader("Books Data")
#     st.dataframe(books)
#     st.subheader("Ratings Data")
#     st.dataframe(ratings)
#     st.subheader("Users Data")
#     st.dataframe(users)

# Enhanced Book Recommender System with Search, Details, Authentication & CSS
