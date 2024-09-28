# samsafront.py
import streamlit as st
import requests

# Streamlit UI Setup
st.set_page_config(
    page_title="Samsa: Text Summarizer & Sentiment Analyzer", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Header Section
st.title("📜 Samsa: Text Summarizer and Sentiment Analyzer")
st.markdown(
    """
    **Upload text files** to summarize their content and analyze their sentiment. 
    Samsa uses advanced AI models to generate concise summaries and sentiment scores.
    """
)
st.write("---")  # Add a horizontal line separator

# File Upload Section
uploaded_files = st.file_uploader(
    "Upload one or more text files", 
    type=["txt"], 
    accept_multiple_files=True,
    help="Supported file format: .txt"
)

# Check if files are uploaded
if uploaded_files:
    # Prepare the files for upload
    files = [('files', (file.name, file, 'text/plain')) for file in uploaded_files]

    with st.spinner("Analyzing... please wait."):
        try:
            # Send the files to the Flask API
            response = requests.post("http://127.0.0.1:5000/analyze", files=files)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse JSON response
            data = response.json()
            st.success("Analysis Complete! Below are the results for each file.")
            st.write("---")

            # Display results for each file
            for result in data:
                # Create a card-like layout using columns
                col1, col2 = st.columns([1, 4])

                with col1:
                    st.subheader(f"📄 {result.get('filename', 'Unknown')}")
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        score = result.get('score', 0)
                        sentiment = "Positive" if score > 66 else "Neutral" if score > 33 else "Negative"
                        sentiment_emoji = "😊" if score > 66 else "😐" if score > 33 else "😞"
                        st.metric(label="Sentiment Score", value=f"{score} / 100", delta=sentiment_emoji)

                with col2:
                    if 'error' not in result:
                        st.write("### 📝 Summary:")
                        st.markdown(f"```{result.get('summary', 'No summary available.')}```")

                st.write("---")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

else:
    # Display a message if no files are uploaded
    st.info("Please upload one or more text files to start the analysis.")
