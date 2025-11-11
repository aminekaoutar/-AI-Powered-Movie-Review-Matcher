from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

app = FastAPI(title="Movie Review Similarity API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vectorizer = None
fightclub_data = None
interstellar_data = None
fightclub_tfidf = None
interstellar_tfidf = None
llm_chain = None

class ReviewRequest(BaseModel):
    movie: str
    user_review: str
    top_n: int = 5
    use_groq_filter: bool = False
    initial_similarity_top_n: int = 20  # How many to get before Groq filtering

class SimilarReview(BaseModel):
    review_id: int
    review_title: str
    review_content: str
    similarity_score: float

def load_and_preprocess_data():
    global vectorizer, fightclub_data, interstellar_data, fightclub_tfidf, interstellar_tfidf, llm_chain
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    try:
        # Load Excel files
        fightclub = pd.read_excel("fightclub_critiques.xlsx")
        interstellar = pd.read_excel("interstellar_critique.xlsx")
        print("✅ Successfully loaded Excel files")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find Excel files. {e}")
        print("📁 Files in current folder:", os.listdir('.'))
        return
    
    # EXACTLY like your Colab code:
    
    # 1. Remove empty rows
    fightclub = fightclub[fightclub["review_content"].notna()]
    fightclub = fightclub[fightclub["review_content"].str.strip() != ""]
    interstellar = interstellar[interstellar["review_content"].notna()]
    interstellar = interstellar[interstellar["review_content"].str.strip() != ""]
    
    # 2. Remove HTML tags
    def remove_html(text):
        if isinstance(text, str):
            return BeautifulSoup(text, "html.parser").get_text()
        return text
    
    fightclub["review_content"] = fightclub["review_content"].apply(remove_html)
    interstellar["review_content"] = interstellar["review_content"].apply(remove_html)
    
    # 3. Ensure string types
    fightclub["review_title"] = fightclub["review_title"].astype(str)
    fightclub["review_content"] = fightclub["review_content"].astype(str)
    interstellar["review_title"] = interstellar["review_title"].astype(str)
    interstellar["review_content"] = interstellar["review_content"].astype(str)
    
    # 4. Combine title + content to create text_to_embed
    fightclub["text_to_embed"] = fightclub["review_title"] + " " + fightclub["review_content"]
    interstellar["text_to_embed"] = interstellar["review_title"] + " " + interstellar["review_content"]
    
    # 5. Add index IDs
    fightclub["index_id"] = range(len(fightclub))
    interstellar["index_id"] = range(len(interstellar))
    
    # 6. Reset index and rename
    fightclub.reset_index(inplace=True)
    fightclub.rename(columns={'index': 'review_id'}, inplace=True)
    interstellar.reset_index(inplace=True)
    interstellar.rename(columns={'index': 'review_id'}, inplace=True)
    
    # Fit TF-IDF and transform
    all_texts = list(fightclub['text_to_embed']) + list(interstellar['text_to_embed'])
    vectorizer.fit(all_texts)
    
    fightclub_tfidf = vectorizer.transform(fightclub['text_to_embed'])
    interstellar_tfidf = vectorizer.transform(interstellar['text_to_embed'])
    
    fightclub_data = fightclub
    interstellar_data = interstellar
    
    # Initialize Groq LLM (like your Colab code)
    try:
        os.environ["GROQ_API_KEY"] = "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"
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        
        # Define the similarity-checking prompt (EXACTLY like your code)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that checks if a review is semantically similar "
                       "to the user's comment. Reply only 'YES' if they express the same topic, "
                       "tone, or sentiment. Otherwise reply 'NO'."),
            ("human", "User comment: {user_input}\nReview: {review_content}")
        ])
        
        # Build the chain
        llm_chain = prompt | llm | StrOutputParser()
        print("✅ Groq LLM initialized successfully")
        
    except Exception as e:
        print(f"⚠️ Groq LLM initialization failed: {e}")
        llm_chain = None
    
    print(f"✅ Data preprocessing completed: {len(fightclub)} Fight Club, {len(interstellar)} Interstellar")

@app.on_event("startup")
async def startup_event():
    load_and_preprocess_data()

def get_similar_reviews(df, tfidf_matrix, user_text, top_n=5):
    user_tfidf = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    similar_reviews = []
    for idx in top_indices:
        row = df.iloc[idx]
        similar_reviews.append(
            SimilarReview(
                review_id=row['review_id'],
                review_title=row['review_title'],
                review_content=row['review_content'],
                similarity_score=float(similarities[idx])
            )
        )
    return similar_reviews

def filter_really_similar_reviews(user_input, similar_reviews):
    """
    Filters reviews that are semantically similar to the user input using Groq LLM.
    Returns a list of review_ids that are truly similar.
    EXACTLY like your Colab code.
    """
    if llm_chain is None:
        print("⚠️ Groq LLM not available, returning all similar reviews")
        return [review.review_id for review in similar_reviews]
    
    really_similar_ids = []

    print("🔍 Checking semantic similarity with Groq LLM...")
    for review in tqdm(similar_reviews):
        try:
            response = llm_chain.invoke({
                "user_input": user_input,
                "review_content": review.review_content
            }).strip().upper()

            if "YES" in response:
                really_similar_ids.append(review.review_id)
        except Exception as e:
            print(f"⚠️ Error checking review {review.review_id}: {e}")
            continue

    return really_similar_ids

@app.get("/")
async def root():
    return {"message": "Movie Review Similarity API"}

@app.post("/find-similar-reviews", response_model=list[SimilarReview])
async def find_similar_reviews(request: ReviewRequest):
    if fightclub_data is None or interstellar_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded properly")
    
    if request.movie.lower() == "fight club":
        df, tfidf_matrix = fightclub_data, fightclub_tfidf
    elif request.movie.lower() == "interstellar":
        df, tfidf_matrix = interstellar_data, interstellar_tfidf
    else:
        raise HTTPException(status_code=400, detail="Movie not supported")
    
    # If Groq filtering is enabled, follow your Colab workflow:
    # 1. First get MORE similar reviews (like top 20 in your Colab)
    # 2. Then use Groq to filter them down
    if request.use_groq_filter and llm_chain is not None:
        print(f"🎯 Using Groq workflow: Get {request.initial_similarity_top_n} similar, then filter with AI")
        
        # Step 1: Get more similar reviews first (like your top_n=20 in Colab)
        initial_similar_reviews = get_similar_reviews(df, tfidf_matrix, request.user_review, request.initial_similarity_top_n)
        
        print(f"📊 Found {len(initial_similar_reviews)} similar reviews before AI filtering")
        
        # Step 2: Apply Groq LLM filtering (like your Colab code)
        really_similar_ids = filter_really_similar_reviews(request.user_review, initial_similar_reviews)
        
        print(f"🤖 After AI filtering: {len(really_similar_ids)} really similar reviews")
        
        # Step 3: Get the final results (limited to requested top_n)
        final_reviews = []
        for review in initial_similar_reviews:
            if review.review_id in really_similar_ids:
                final_reviews.append(review)
            if len(final_reviews) >= request.top_n:  # Don't exceed requested number
                break
        
        # If no reviews passed Groq filter, return the top TF-IDF results as fallback
        if not final_reviews:
            print("⚠️ No reviews passed Groq filter, returning top TF-IDF results")
            return get_similar_reviews(df, tfidf_matrix, request.user_review, request.top_n)
        
        return final_reviews
    
    else:
        # Normal TF-IDF only workflow
        print(f"🔍 Using TF-IDF only: Get {request.top_n} similar reviews")
        return get_similar_reviews(df, tfidf_matrix, request.user_review, request.top_n)

@app.get("/movies")
async def get_available_movies():
    return {"movies": ["Fight Club", "Interstellar"]}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    groq_status = "available" if llm_chain is not None else "unavailable"
    return {
        "status": "healthy",
        "fightclub_reviews": len(fightclub_data) if fightclub_data is not None else 0,
        "interstellar_reviews": len(interstellar_data) if interstellar_data is not None else 0,
        "groq_llm": groq_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)