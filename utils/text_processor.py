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

def simple_tokenize(text):
    """Simple tokenizer that splits on whitespace and punctuation"""
    # Replace punctuation with spaces and then split
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

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
        # Check that the text has enough content to process
        if not text or len(text.strip()) < 10:
            logging.warning("Input text is too short for meaningful processing")
            return [], []
            
        # Split text into sentences using NLTK, or fallback to basic splitting
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logging.warning(f"NLTK sentence tokenization failed, using fallback: {str(e)}")
            # Basic sentence splitting on periods, question marks, and exclamation points
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            logging.warning("Could not identify any sentences in the text")
            return [], []
            
        # Get stopwords (with fallback to common English stopwords)
        try:
            stop_words = set(stopwords.words('english'))
        except Exception as e:
            logging.warning(f"Could not get NLTK stopwords, using fallback list: {str(e)}")
            # Common English stopwords as fallback
            stop_words = set([
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
                'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
                't', 'can', 'will', 'just', 'don', 'should', 'now'
            ])
        
        # Initialize list for tokens
        all_tokens = []
        
        # Process each sentence
        for sentence in sentences:
            # Convert to lowercase
            sentence = sentence.lower()
            
            # Remove punctuation except periods (to keep sentence structure)
            sentence = re.sub(r'[^\w\s\.]', '', sentence)
            
            # Try to tokenize with NLTK, fall back to simple tokenize if it fails
            try:
                tokens = word_tokenize(sentence)
            except Exception as e:
                logging.warning(f"NLTK tokenization failed, using fallback: {str(e)}")
                tokens = simple_tokenize(sentence)
            
            # Remove stopwords and non-alphabetic tokens
            tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
            
            # Add to the list of all tokens
            all_tokens.extend(tokens)
        
        # Check if we have enough tokens for meaningful analysis
        if len(all_tokens) < 5:
            logging.warning("Not enough meaningful words found after preprocessing")
            return [], []
            
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
            # Lowercase the sentence
            sentence_lower = sentence.lower()
            
            # Try to tokenize with NLTK, fall back to simple tokenize if it fails
            try:
                words = word_tokenize(sentence_lower)
            except Exception as e:
                logging.warning(f"NLTK tokenization failed in summary generation, using fallback: {str(e)}")
                words = simple_tokenize(sentence_lower)
            
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
