import os
import torch
import logging
from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langchain_ollama.llms import OllamaLLM

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Check for CUDA availability
if torch.cuda.is_available():
    logging.info(f"CUDA Version: {torch.version.cuda}")
else:
    logging.warning("CUDA is not available.")

# Load the sentiment analysis model
SENTIMENT_MODEL_NAME = "siebert/sentiment-roberta-large-english"
try:
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME, use_fast=False, clean_up_tokenization_spaces=True)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
except Exception as e:
    logging.error(f"Error loading sentiment analysis model: {str(e)}")
    model = None  # Set to None to handle failed loading

# Initialize sentiment analysis pipeline
if model:
    sentiment_analysis = pipeline(
        "sentiment-analysis", 
        model=model, 
        tokenizer=tokenizer, 
        device=0 if torch.cuda.is_available() else -1
    )

# Initialize LLaMA model from Ollama using LangChain
llm = OllamaLLM(model="llama3.1", temperature=0.1)

def llama_summarize(text):
    """Summarizes the input text using LLaMA 3.1 8B via LangChain and Ollama."""
    prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
    try:
        summary = llm.invoke(prompt)
        return summary.strip() if summary else "Summarization failed."
    except Exception as e:
        logging.error(f"Error during text summarization: {str(e)}")
        return "Summarization failed."

def score_sentiment(sentiment):
    """Converts sentiment label to a score (1-100 scale)."""
    label_to_score = {
        "positive": 80,
        "neutral": 50,
        "negative": 20
    }
    return label_to_score.get(sentiment, 50)

@app.route('/analyze', methods=['POST'])
def analyze_text_files():
    """API endpoint to analyze multiple text files: summarizes using LLaMA and performs sentiment analysis."""
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files uploaded."}), 400
    
    results = []

    for file in files:
        try:
            text = file.read().decode('utf-8')
            if not text.strip():
                results.append({"filename": file.filename, "error": "The file is empty or unreadable."})
                continue
        except Exception as e:
            logging.error(f"Error reading the uploaded file {file.filename}: {str(e)}")
            results.append({"filename": file.filename, "error": "Failed to read the file content."})
            continue
        
        # Normalize text
        text = text.lower().strip()

        # Step 1: Summarize the text using LLaMA
        summary = llama_summarize(text)
        if summary == "Summarization failed.":
            results.append({"filename": file.filename, "error": "Failed to summarize the text."})
            continue
        
        # Step 2: Perform sentiment analysis on the summary
        if model:
            try:
                result = sentiment_analysis(summary[:512])  # Truncate input length to 512 tokens
                if not result:
                    raise ValueError("Sentiment analysis returned an empty result.")

                # Log the result for debugging
                logging.info(f"Sentiment analysis result for {file.filename}: {result}")

                # Extract label and score
                sentiment_label = result[0].get('label', '').lower()
                sentiment_score = result[0].get('score', 0)  # Ensure the score is extracted
                
                # Validate sentiment label
                if sentiment_label not in ['positive', 'neutral', 'negative']:
                    raise ValueError(f"Unexpected sentiment label: {sentiment_label}")

                overall_score = score_sentiment(sentiment_label)
                results.append({"filename": file.filename, "summary": summary, "score": overall_score, "sentiment_label": sentiment_label, "sentiment_score": sentiment_score})
            except Exception as e:
                logging.error(f"Error during sentiment analysis of file {file.filename}: {str(e)}")
                results.append({"filename": file.filename, "error": "Failed during sentiment analysis."})
        else:
            logging.error("Sentiment analysis model is not available.")
            results.append({"filename": file.filename, "error": "Sentiment analysis model failed to load."})

    return jsonify(results), 200


if __name__ == '__main__':
    app.run(debug=True)
