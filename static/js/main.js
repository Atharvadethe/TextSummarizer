document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('summarize-form');
    const inputText = document.getElementById('input-text');
    const numSentences = document.getElementById('num-sentences');
    const summarySection = document.getElementById('summary-section');
    const summaryContent = document.getElementById('summary-content');
    const keywordsSection = document.getElementById('keywords-section');
    const keywordsList = document.getElementById('keywords-list');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorAlert = document.getElementById('error-alert');
    
    // Initially hide the results and error sections
    summarySection.style.display = 'none';
    keywordsSection.style.display = 'none';
    errorAlert.style.display = 'none';
    loadingSpinner.style.display = 'none';
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Get the input text
        const text = inputText.value.trim();
        const sentences = parseInt(numSentences.value);
        
        // Validate input
        if (!text) {
            showError('Please enter some text to summarize.');
            return;
        }
        
        // Show loading spinner
        loadingSpinner.style.display = 'block';
        errorAlert.style.display = 'none';
        summarySection.style.display = 'none';
        keywordsSection.style.display = 'none';
        
        try {
            // Send the request to the server
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    num_sentences: sentences
                })
            });
            
            // Parse the response
            const data = await response.json();
            
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            if (response.ok) {
                // Display the summary
                summaryContent.textContent = data.summary;
                summarySection.style.display = 'block';
                
                // Display the keywords
                keywordsList.innerHTML = '';
                if (data.top_keywords && data.top_keywords.length > 0) {
                    data.top_keywords.forEach(function(item) {
                        const keyword = item[0];
                        const score = item[1].toFixed(4);
                        const li = document.createElement('li');
                        li.className = 'list-group-item d-flex justify-content-between align-items-center';
                        li.innerHTML = `
                            <span>${keyword}</span>
                            <span class="badge bg-primary rounded-pill">${score}</span>
                        `;
                        keywordsList.appendChild(li);
                    });
                    keywordsSection.style.display = 'block';
                }
                
                // Scroll to results
                summarySection.scrollIntoView({ behavior: 'smooth' });
            } else {
                showError(data.error || 'An error occurred while processing your text.');
            }
        } catch (error) {
            console.error('Error:', error);
            loadingSpinner.style.display = 'none';
            showError('Network error: Could not connect to the server.');
        }
    });
    
    function showError(message) {
        errorAlert.textContent = message;
        errorAlert.style.display = 'block';
        loadingSpinner.style.display = 'none';
        errorAlert.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Add example text function
    document.getElementById('add-example').addEventListener('click', function() {
        inputText.value = "Text summarization is the process of distilling the most important information from a source text. There are two main approaches to text summarization: extractive and abstractive. Extractive summarization involves selecting important sentences from the original text to form a summary. Abstractive summarization involves generating new sentences that capture the meaning of the original text. TF-IDF, which stands for Term Frequency-Inverse Document Frequency, is a numerical statistic used in information retrieval to reflect how important a word is to a document in a collection. In the context of text summarization, TF-IDF can be used to identify keywords that are significant to the document.";
    });
    
    // Clear text function
    document.getElementById('clear-text').addEventListener('click', function() {
        inputText.value = '';
        summarySection.style.display = 'none';
        keywordsSection.style.display = 'none';
        errorAlert.style.display = 'none';
    });
});
