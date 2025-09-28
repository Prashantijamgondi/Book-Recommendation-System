# Enhanced Search and Preprocessing for Book Recommender System
# Run this script to create additional models for better search functionality

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class EnhancedBookSearch:
    def __init__(self, books_df):
        self.books_df = books_df
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.processed_books = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text for better search"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def prepare_search_data(self):
        """Prepare data for enhanced search"""
        # Create combined text for search
        self.books_df['search_text'] = (
            self.books_df['Book-Title'].fillna('') + ' ' + 
            self.books_df['Book-Author'].fillna('') + ' ' + 
            self.books_df['Publisher'].fillna('')
        )
        
        # Preprocess the search text
        self.books_df['processed_search'] = self.books_df['search_text'].apply(self.preprocess_text)
        
        # Create TF-IDF vectors for content-based search
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.books_df['processed_search'])
        
        return self
    
    def fuzzy_search(self, query, threshold=60):
        """Perform fuzzy string matching search"""
        query_lower = query.lower()
        results = []
        
        for idx, row in self.books_df.iterrows():
            title_score = fuzz.partial_ratio(query_lower, str(row['Book-Title']).lower())
            author_score = fuzz.partial_ratio(query_lower, str(row['Book-Author']).lower())
            
            max_score = max(title_score, author_score)
            
            if max_score >= threshold:
                results.append({
                    'index': idx,
                    'score': max_score,
                    'book': row
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:20]  # Return top 20 results
    
    def semantic_search(self, query, top_k=20):
        """Perform semantic search using TF-IDF"""
        if self.tfidf_matrix is None:
            return []
        
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append({
                    'index': idx,
                    'score': similarities[idx],
                    'book': self.books_df.iloc[idx]
                })
        
        return results
    
    def combined_search(self, query, limit=20):
        """Combine fuzzy and semantic search for better results"""
        fuzzy_results = self.fuzzy_search(query)
        semantic_results = self.semantic_search(query)
        
        # Combine and deduplicate results
        combined_results = {}
        
        # Add fuzzy results with weight
        for result in fuzzy_results:
            idx = result['index']
            combined_results[idx] = {
                'book': result['book'],
                'score': result['score'] * 0.01,  # Normalize fuzzy score
                'type': 'fuzzy'
            }
        
        # Add semantic results with weight
        for result in semantic_results:
            idx = result['index']
            if idx in combined_results:
                # Combine scores if book found by both methods
                combined_results[idx]['score'] += result['score']
                combined_results[idx]['type'] = 'combined'
            else:
                combined_results[idx] = {
                    'book': result['book'],
                    'score': result['score'],
                    'type': 'semantic'
                }
        
        # Sort by combined score
        sorted_results = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)
        
        return sorted_results[:limit]

def create_enhanced_models():
    """Create and save enhanced search models"""
    
    # Load your existing books data
    try:
        books = pickle.load(open('Models/books.pkl', 'rb'))
        print("Loaded existing books data")
    except FileNotFoundError:
        print("Books.pkl not found. Please ensure the file exists.")
        return
    
    # Create enhanced search object
    search_engine = EnhancedBookSearch(books)
    search_engine.prepare_search_data()
    
    # Save the enhanced search models
    with open('Models/enhanced_search.pkl', 'wb') as f:
        pickle.dump(search_engine, f)
    
    print("Enhanced search model saved successfully!")
    
    # Create genre-based recommendations if genre data is available
    if 'Genre' in books.columns:
        create_genre_recommendations(books)
    
    # Create author-based recommendations
    create_author_recommendations(books)

def create_genre_recommendations(books_df):
    """Create genre-based recommendation system"""
    # This assumes you have genre information
    genre_books = {}
    
    for idx, row in books_df.iterrows():
        genres = str(row.get('Genre', '')).split(',')
        for genre in genres:
            genre = genre.strip().lower()
            if genre and genre != 'nan':
                if genre not in genre_books:
                    genre_books[genre] = []
                genre_books[genre].append(row)
    
    # Save genre recommendations
    with open('Models/genre_recommendations.pkl', 'wb') as f:
        pickle.dump(genre_books, f)
    
    print("Genre recommendations saved successfully!")

def create_author_recommendations(books_df):
    """Create author-based recommendation system"""
    author_books = {}
    
    for idx, row in books_df.iterrows():
        author = str(row['Book-Author']).strip()
        if author and author != 'nan':
            if author not in author_books:
                author_books[author] = []
            author_books[author].append(row)
    
    # Save author recommendations
    with open('Models/author_recommendations.pkl', 'wb') as f:
        pickle.dump(author_books, f)
    
    print("Author recommendations saved successfully!")

def create_user_profile_system():
    """Create user profile and history tracking system"""
    
    class UserProfileSystem:
        def __init__(self):
            self.user_profiles = {}
            self.user_history = {}
        
        def add_user_interaction(self, username, book_title, interaction_type):
            """Track user interactions (view, like, rate, etc.)"""
            if username not in self.user_history:
                self.user_history[username] = []
            
            self.user_history[username].append({
                'book_title': book_title,
                'interaction_type': interaction_type,
                'timestamp': pd.Timestamp.now()
            })
        
        def get_user_recommendations(self, username, books_df, similarity_scores, pt):
            """Get personalized recommendations based on user history"""
            if username not in self.user_history:
                return []
            
            user_interactions = self.user_history[username]
            
            # Get books user has interacted with
            user_books = [interaction['book_title'] for interaction in user_interactions]
            
            # Find similar books to user's history
            recommendations = set()
            
            for book in user_books:
                if book in pt.index:
                    try:
                        book_idx = np.where(pt.index == book)[0][0]
                        similar_books = sorted(
                            list(enumerate(similarity_scores[book_idx])), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[1:6]  # Get top 5 similar books
                        
                        for similar_book_idx, score in similar_books:
                            similar_book_title = pt.index[similar_book_idx]
                            if similar_book_title not in user_books:
                                recommendations.add(similar_book_title)
                    except:
                        continue
            
            return list(recommendations)[:10]  # Return top 10 recommendations
    
    # Save user profile system
    user_system = UserProfileSystem()
    with open('Models/user_profile_system.pkl', 'wb') as f:
        pickle.dump(user_system, f)
    
    print("User profile system saved successfully!")

# Advanced filtering system
def create_advanced_filters():
    """Create advanced filtering options"""
    
    class AdvancedFilters:
        def __init__(self, books_df):
            self.books_df = books_df
            self.prepare_filter_data()
        
        def prepare_filter_data(self):
            """Prepare data for filtering"""
            # Clean year data
            self.books_df['Year-Of-Publication'] = pd.to_numeric(
                self.books_df['Year-Of-Publication'], 
                errors='coerce'
            )
            
            # Create decade categories
            self.books_df['Decade'] = (self.books_df['Year-Of-Publication'] // 10) * 10
            
            # Extract unique values for filters
            self.authors = sorted(self.books_df['Book-Author'].dropna().unique())
            self.publishers = sorted(self.books_df['Publisher'].dropna().unique())
            self.decades = sorted(self.books_df['Decade'].dropna().unique())
        
        def filter_books(self, author=None, publisher=None, year_range=None, 
                        decade=None, rating_range=None, title_contains=None):
            """Apply multiple filters to book dataset"""
            filtered_df = self.books_df.copy()
            
            if author:
                filtered_df = filtered_df[
                    filtered_df['Book-Author'].str.contains(author, case=False, na=False)
                ]
            
            if publisher:
                filtered_df = filtered_df[
                    filtered_df['Publisher'].str.contains(publisher, case=False, na=False)
                ]
            
            if year_range:
                start_year, end_year = year_range
                filtered_df = filtered_df[
                    (filtered_df['Year-Of-Publication'] >= start_year) & 
                    (filtered_df['Year-Of-Publication'] <= end_year)
                ]
            
            if decade:
                filtered_df = filtered_df[filtered_df['Decade'] == decade]
            
            if rating_range and 'Book-Rating' in filtered_df.columns:
                min_rating, max_rating = rating_range
                filtered_df = filtered_df[
                    (filtered_df['Book-Rating'] >= min_rating) & 
                    (filtered_df['Book-Rating'] <= max_rating)
                ]
            
            if title_contains:
                filtered_df = filtered_df[
                    filtered_df['Book-Title'].str.contains(title_contains, case=False, na=False)
                ]
            
            return filtered_df
    
    # Load books data
    try:
        books = pickle.load(open('Models/books.pkl', 'rb'))
        filter_system = AdvancedFilters(books)
        
        # Save advanced filter system
        with open('Models/advanced_filters.pkl', 'wb') as f:
            pickle.dump(filter_system, f)
        
        print("Advanced filters saved successfully!")
        
    except FileNotFoundError:
        print("Books.pkl not found for creating advanced filters.")

def create_book_analytics():
    """Create analytics and insights system"""
    
    class BookAnalytics:
        def __init__(self, books_df):
            self.books_df = books_df
            self.prepare_analytics_data()
        
        def prepare_analytics_data(self):
            """Prepare data for analytics"""
            # Clean and prepare numeric data
            self.books_df['Year-Of-Publication'] = pd.to_numeric(
                self.books_df['Year-Of-Publication'], 
                errors='coerce'
            )
            
            if 'Book-Rating' in self.books_df.columns:
                self.books_df['Book-Rating'] = pd.to_numeric(
                    self.books_df['Book-Rating'], 
                    errors='coerce'
                )
        
        def get_top_authors(self, limit=20):
            """Get top authors by book count"""
            author_counts = self.books_df['Book-Author'].value_counts().head(limit)
            return author_counts.to_dict()
        
        def get_top_publishers(self, limit=20):
            """Get top publishers by book count"""
            publisher_counts = self.books_df['Publisher'].value_counts().head(limit)
            return publisher_counts.to_dict()
        
        def get_publication_trends(self):
            """Get publication trends by year"""
            yearly_counts = self.books_df.groupby('Year-Of-Publication').size()
            return yearly_counts.to_dict()
        
        def get_rating_distribution(self):
            """Get rating distribution"""
            if 'Book-Rating' in self.books_df.columns:
                rating_dist = self.books_df['Book-Rating'].value_counts().sort_index()
                return rating_dist.to_dict()
            return {}
        
        def get_book_stats(self):
            """Get general book statistics"""
            stats = {
                'total_books': len(self.books_df),
                'total_authors': self.books_df['Book-Author'].nunique(),
                'total_publishers': self.books_df['Publisher'].nunique(),
                'year_range': (
                    int(self.books_df['Year-Of-Publication'].min()) if pd.notna(self.books_df['Year-Of-Publication'].min()) else None,
                    int(self.books_df['Year-Of-Publication'].max()) if pd.notna(self.books_df['Year-Of-Publication'].max()) else None
                )
            }
            
            if 'Book-Rating' in self.books_df.columns:
                stats['avg_rating'] = self.books_df['Book-Rating'].mean()
                stats['rating_range'] = (
                    self.books_df['Book-Rating'].min(),
                    self.books_df['Book-Rating'].max()
                )
            
            return stats
    
    # Load books data and create analytics
    try:
        books = pickle.load(open('Models/books.pkl', 'rb'))
        analytics_system = BookAnalytics(books)
        
        # Save analytics system
        with open('Models/book_analytics.pkl', 'wb') as f:
            pickle.dump(analytics_system, f)
        
        print("Book analytics saved successfully!")
        
    except FileNotFoundError:
        print("Books.pkl not found for creating analytics.")

def create_recommendation_explanations():
    """Create system to explain why books are recommended"""
    
    class RecommendationExplainer:
        def __init__(self, books_df, similarity_scores, pt):
            self.books_df = books_df
            self.similarity_scores = similarity_scores
            self.pt = pt
        
        def explain_recommendation(self, source_book, recommended_book):
            """Explain why a book is recommended"""
            explanations = []
            
            # Get book details
            source_data = self.books_df[self.books_df['Book-Title'] == source_book]
            rec_data = self.books_df[self.books_df['Book-Title'] == recommended_book]
            
            if source_data.empty or rec_data.empty:
                return ["Books not found in database"]
            
            source_book_data = source_data.iloc[0]
            rec_book_data = rec_data.iloc[0]
            
            # Check author similarity
            if source_book_data['Book-Author'] == rec_book_data['Book-Author']:
                explanations.append(f"Same author: {source_book_data['Book-Author']}")
            
            # Check publisher similarity
            if source_book_data['Publisher'] == rec_book_data['Publisher']:
                explanations.append(f"Same publisher: {source_book_data['Publisher']}")
            
            # Check publication year proximity
            try:
                source_year = int(source_book_data['Year-Of-Publication'])
                rec_year = int(rec_book_data['Year-Of-Publication'])
                year_diff = abs(source_year - rec_year)
                
                if year_diff <= 5:
                    explanations.append(f"Published around the same time (within {year_diff} years)")
            except:
                pass
            
            # Get similarity score
            try:
                source_idx = np.where(self.pt.index == source_book)[0][0]
                rec_idx = np.where(self.pt.index == recommended_book)[0][0]
                similarity = self.similarity_scores[source_idx][rec_idx]
                
                if similarity > 0.8:
                    explanations.append("Very high user preference similarity")
                elif similarity > 0.6:
                    explanations.append("High user preference similarity")
                elif similarity > 0.4:
                    explanations.append("Moderate user preference similarity")
            except:
                pass
            
            if not explanations:
                explanations.append("Recommended based on user reading patterns")
            
            return explanations
        
        def get_recommendation_strength(self, source_book, recommended_book):
            """Get recommendation strength score"""
            try:
                source_idx = np.where(self.pt.index == source_book)[0][0]
                rec_idx = np.where(self.pt.index == recommended_book)[0][0]
                similarity = self.similarity_scores[source_idx][rec_idx]
                
                if similarity > 0.8:
                    return "Very Strong"
                elif similarity > 0.6:
                    return "Strong"
                elif similarity > 0.4:
                    return "Moderate"
                else:
                    return "Weak"
            except:
                return "Unknown"
    
    # Load required data and create explainer
    try:
        books = pickle.load(open('Models/books.pkl', 'rb'))
        similarity_scores = pickle.load(open('Models/similarity_scores.pkl', 'rb'))
        pt = pickle.load(open('Models/pt.pkl', 'rb'))
        
        explainer = RecommendationExplainer(books, similarity_scores, pt)
        
        # Save recommendation explainer
        with open('Models/recommendation_explainer.pkl', 'wb') as f:
            pickle.dump(explainer, f)
        
        print("Recommendation explainer saved successfully!")
        
    except FileNotFoundError:
        print("Required model files not found for creating explainer.")

def create_trending_books():
    """Create trending books system based on recent interactions"""
    
    class TrendingBooks:
        def __init__(self, books_df):
            self.books_df = books_df
            self.interaction_log = []
        
        def log_interaction(self, book_title, interaction_type="view"):
            """Log book interactions"""
            self.interaction_log.append({
                'book_title': book_title,
                'interaction_type': interaction_type,
                'timestamp': pd.Timestamp.now()
            })
        
        def get_trending_books(self, days=7, limit=20):
            """Get trending books based on recent interactions"""
            if not self.interaction_log:
                # Return random sample if no interactions
                return self.books_df.sample(n=min(limit, len(self.books_df)))
            
            # Filter interactions from last N days
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            recent_interactions = [
                interaction for interaction in self.interaction_log 
                if interaction['timestamp'] >= cutoff_date
            ]
            
            if not recent_interactions:
                return self.books_df.sample(n=min(limit, len(self.books_df)))
            
            # Count interactions per book
            interaction_counts = {}
            for interaction in recent_interactions:
                book_title = interaction['book_title']
                interaction_counts[book_title] = interaction_counts.get(book_title, 0) + 1
            
            # Sort by interaction count
            trending_titles = sorted(
                interaction_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            # Get book details for trending titles
            trending_books = []
            for title, count in trending_titles:
                book_data = self.books_df[self.books_df['Book-Title'] == title]
                if not book_data.empty:
                    trending_books.append(book_data.iloc[0])
            
            return pd.DataFrame(trending_books)
    
    # Load books data and create trending system
    try:
        books = pickle.load(open('Models/books.pkl', 'rb'))
        trending_system = TrendingBooks(books)
        
        # Save trending books system
        with open('Models/trending_books.pkl', 'wb') as f:
            pickle.dump(trending_system, f)
        
        print("Trending books system saved successfully!")
        
    except FileNotFoundError:
        print("Books.pkl not found for creating trending system.")

# Main execution
if __name__ == "__main__":
    print("Creating enhanced models for book recommender system...")
    print("=" * 60)
    
    # Create all enhanced models
    create_enhanced_models()
    create_user_profile_system()
    create_advanced_filters()
    create_book_analytics()
    create_recommendation_explanations()
    create_trending_books()
    
    print("=" * 60)
    print("All enhanced models created successfully!")
    print("\nFiles created:")
    print("- Models/enhanced_search.pkl")
    print("- Models/user_profile_system.pkl")
    print("- Models/advanced_filters.pkl")
    print("- Models/book_analytics.pkl")
    print("- Models/recommendation_explainer.pkl")
    print("- Models/trending_books.pkl")
    print("- Models/genre_recommendations.pkl (if genre data available)")
    print("- Models/author_recommendations.pkl")
    
    print("\nTo install required dependencies, run:")
    print("pip install fuzzywuzzy nltk scikit-learn")
    
    print("\nYour enhanced book recommender system is ready!")
