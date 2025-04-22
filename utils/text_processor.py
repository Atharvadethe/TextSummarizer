import re
import nltk
import logging
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download necessary NLTK packages
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def preprocess_text(text):
    """
    Preprocess the text: lowercase, remove punctuation, remove stopwords, tokenize.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        tuple: (tokens, sentences) - list of tokens and list of sentences
    """
    try:
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Get stopwords
        stop_words = set(stopwords.words('english'))
        
        # Initialize list for tokens
        all_tokens = []
        
        # Process each sentence
        for sentence in sentences:
            # Convert to lowercase
            sentence = sentence.lower()
            
            # Remove punctuation except periods (to keep sentence structure)
            sentence = re.sub(r'[^\w\s\.]', '', sentence)
            
            # Tokenize the sentence
            tokens = word_tokenize(sentence)
            
            # Remove stopwords and non-alphabetic tokens
            tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
            
            # Add to the list of all tokens
            all_tokens.extend(tokens)
        
        return all_tokens, sentences
    except Exception as e:
        logging.error(f"Error preprocessing text: {str(e)}")
        return [], []

def calculate_tf_idf(tokens, sentences):
    """
    Calculate TF-IDF scores for each word in the text.
    
    Args:
        tokens (list): List of preprocessed tokens
        sentences (list): List of sentences from the original text
        
    Returns:
        dict: Dictionary mapping words to their TF-IDF scores
    """
    try:
        # Calculate term frequency (TF)
        tf = Counter(tokens)
        
        # Calculate document frequency (DF)
        df = {}
        for word in set(tokens):
            df[word] = sum(1 for sentence in sentences if word.lower() in sentence.lower())
        
        # Calculate inverse document frequency (IDF)
        num_sentences = len(sentences)
        idf = {word: np.log(num_sentences / (1 + df[word])) for word in df}
        
        # Calculate TF-IDF
        tf_idf = {word: tf[word] * idf[word] for word in tf}
        
        return tf_idf
    except Exception as e:
        logging.error(f"Error calculating TF-IDF: {str(e)}")
        return {}

def get_top_keywords(tf_idf_scores, num_keywords=5):
    """
    Get the top N keywords based on TF-IDF scores.
    
    Args:
        tf_idf_scores (dict): Dictionary mapping words to their TF-IDF scores
        num_keywords (int): Number of top keywords to return
        
    Returns:
        list: List of tuples (keyword, score) sorted by score
    """
    try:
        # Sort words by TF-IDF score in descending order
        sorted_words = sorted(tf_idf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N keywords
        return sorted_words[:num_keywords]
    except Exception as e:
        logging.error(f"Error getting top keywords: {str(e)}")
        return []

def generate_summary(sentences, tf_idf_scores, num_sentences=3):
    """
    Generate an extractive summary by selecting sentences with the highest keyword scores.
    
    Args:
        sentences (list): List of sentences from the original text
        tf_idf_scores (dict): Dictionary mapping words to their TF-IDF scores
        num_sentences (int): Number of sentences to include in the summary
        
    Returns:
        str: Summary text
    """
    try:
        # Calculate score for each sentence
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            # Lowercase and tokenize the sentence
            words = word_tokenize(sentence.lower())
            
            # Calculate sentence score based on TF-IDF scores of words
            score = sum(tf_idf_scores.get(word, 0) for word in words)
            
            # Normalize by sentence length to avoid bias towards longer sentences
            # Add a small constant to avoid division by zero
            score = score / (len(words) + 0.1)
            
            # Store the sentence index and its score
            sentence_scores.append((i, score))
        
        # Sort sentences by score in descending order
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        
        # Get the indices of top-scoring sentences
        top_indices = [idx for idx, _ in sorted_sentences[:num_sentences]]
        
        # Sort the indices to maintain original order
        top_indices.sort()
        
        # Join the top sentences to form the summary
        summary = ' '.join([sentences[i] for i in top_indices])
        
        return summary
    except Exception as e:
        logging.error(f"Error generating summary: {str(e)}")
        return "Could not generate summary."
