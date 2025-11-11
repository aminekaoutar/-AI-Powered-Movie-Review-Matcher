import streamlit as st
import requests
import json

# Configuration
API_BASE_URL = "http://localhost:8000"

def main():
    st.set_page_config(
        page_title="Movie Review Matcher",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Movie Review Similarity Finder")
    st.markdown("""
    Find reviews that are similar to your thoughts about a movie!
    Simply select a movie and type your review to see the most similar existing reviews.
    """)
    
    # Sidebar for movie selection and options
    st.sidebar.header("Movie Selection")
    movie_choice = st.sidebar.radio(
        "Choose a movie:",
        ["Fight Club", "Interstellar"],
        help="Select the movie you want to analyze reviews for"
    )
    
    # Groq LLM filter option
    use_groq_filter = st.sidebar.checkbox(
        "Use AI Semantic Filter (Groq)",
        value=False,
        help="Use AI to find only reviews that are semantically similar to your comment"
    )
    
    # Advanced options (collapsible)
    with st.sidebar.expander("Advanced Options"):
        if use_groq_filter:
            initial_search_count = st.slider(
                "Initial search count (before AI filtering):",
                min_value=10,
                max_value=50,
                value=20,
                help="More reviews = better AI filtering but slower"
            )
        else:
            initial_search_count = 20  # Default
        
        final_results_count = st.slider(
            "Final results to show:",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of reviews to display"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"Your Review for {movie_choice}")
        
        # User input
        user_review = st.text_area(
            "Type your review here:",
            height=200,
            placeholder=f"Share your thoughts about {movie_choice}...",
            help="Be as detailed as possible for better matches"
        )
        
        # Find similar reviews button
        if st.button("Find Similar Reviews", type="primary"):
            if not user_review.strip():
                st.error("Please enter a review before searching.")
            else:
                with st.spinner(f"Finding similar reviews for {movie_choice}..."):
                    try:
                        # Call FastAPI backend
                        response = requests.post(
                            f"{API_BASE_URL}/find-similar-reviews",
                            json={
                                "movie": movie_choice,
                                "user_review": user_review,
                                "top_n": final_results_count,
                                "use_groq_filter": use_groq_filter,
                                "initial_similarity_top_n": initial_search_count if use_groq_filter else final_results_count
                            }
                        )
                        
                        if response.status_code == 200:
                            similar_reviews = response.json()
                            display_results(similar_reviews, col2, use_groq_filter)
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to the server. Make sure the FastAPI backend is running on port 8000.")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
    
    # Instructions in sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("How It Works")
        
        if use_groq_filter:
            st.markdown("""
            **🤖 With AI Filter:**
            1. Find **{} reviews** using similarity search
            2. AI checks each one for **semantic meaning**
            3. Show **top {}** that pass AI check
            """.format(initial_search_count, final_results_count))
        else:
            st.markdown("""
            **🔍 Without AI Filter:**
            1. Find **{} reviews** using similarity search
            2. Show results directly
            """.format(final_results_count))

def display_results(similar_reviews, column, use_groq_filter=False):
    """Display the similar reviews in a formatted way"""
    with column:
        if use_groq_filter:
            st.subheader("🤖 AI-Filtered Similar Reviews")
            st.info(f"Showing {len(similar_reviews)} reviews that passed AI semantic similarity check")
        else:
            st.subheader("🔍 Similar Reviews Found")
            st.info(f"Showing {len(similar_reviews)} most similar reviews")
        
        if not similar_reviews:
            st.info("No similar reviews found. Try being more specific in your review.")
            return
        
        for i, review in enumerate(similar_reviews, 1):
            with st.container():
                st.markdown(f"### Match #{i} (Similarity: {review['similarity_score']:.2f})")
                
                # Review title
                if review['review_title'] and review['review_title'] != 'nan':
                    st.markdown(f"**Title:** {review['review_title']}")
                
                # Review content
                st.markdown("**Review:**")
                st.text_area(
                    f"Review {i} content",
                    value=review['review_content'],
                    height=150,
                    key=f"review_{i}_{use_groq_filter}",  # Unique key for each mode
                    label_visibility="collapsed"
                )
                
                st.markdown("---")

if __name__ == "__main__":
    main()