import os
import logging
from flask import Flask, render_template, request, jsonify
from utils.text_processor import preprocess_text, calculate_tf_idf, get_top_keywords, generate_summary

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

@app.route('/')
def index():
    """Render the main page of the application."""
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    """Process text and return summary with keywords."""
    try:
        # Get text from the request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
            
        # Get summary sentences count from request or default to 3
        num_sentences = int(data.get('num_sentences', 3))
        
        # Process the text
        tokens, sentences = preprocess_text(text)
        
        if not tokens or not sentences:
            return jsonify({'error': 'Could not process the text. Please provide more content with complete sentences (at least a paragraph with 2-3 sentences).'}), 400
            
        # Calculate TF-IDF
        tf_idf_scores = calculate_tf_idf(tokens, sentences)
        
        # Get top keywords
        top_keywords = get_top_keywords(tf_idf_scores, 5)
        
        # Generate summary
        summary = generate_summary(sentences, tf_idf_scores, num_sentences)
        
        # Return the results
        return jsonify({
            'summary': summary,
            'top_keywords': top_keywords
        })
    
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}")
        return jsonify({'error': f'Error processing text: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
