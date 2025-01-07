import streamlit as st
import chromadb
import ollama
import json
import pandas as pd
import re

# Initialize ChromaDB client
client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_or_create_collection(name="Store_Inventory")

# Sidebar with instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Enter your question or search query in the input box.
2. Adjust the number of results using the slider.
3. View matching products and their details below.
""")
st.sidebar.markdown("---")

# Main UI
st.title("🛒 Chat with Store Bot")

# User input
user_input = st.text_input("🔍 Ask a question about a product:")
n_results = st.slider("Number of results to display:", min_value=1, max_value=20, value=10)

# Function to validate input
def is_valid_input(text):
    if len(text) < 3:  # Too short
        return False, "Query is too short. Please enter a more descriptive question."
    if re.search(r"(.)\1{4,}", text):  # Repeated characters
        return False, "Query contains repeated characters or nonsense text."
    return True, None

# Function to filter results based on query
def filter_results(query, results):
    # Parse documents
    documents = results['documents']
    if not documents or len(documents[0]) == 0:
        return None, "No matching products found."
    
    parsed_documents = [json.loads(doc) for doc in documents[0]]
    df = pd.DataFrame(parsed_documents)
    
    # Identify what the user is asking for
    if "aisle" in query.lower():
        # Return aisle information
        filtered_df = df[["Product", "Aisle Name", "Aisle Number"]]
        return filtered_df, None
    elif "price" in query.lower():
        # Return price information
        filtered_df = df[["Product", "Price"]]
        return filtered_df, None
    else:
        # Default: return all product details
        return df, None

# Validate and process user input
if user_input:
    is_valid, error_message = is_valid_input(user_input)
    if not is_valid:
        st.warning(error_message)
    else:
        try:
            # Generate embedding for the query
            response = ollama.embeddings(
                prompt=user_input,
                model="mxbai-embed-large"
            )

            # Query ChromaDB
            results = collection.query(
                query_embeddings=[response["embedding"]],
                n_results=n_results
            )

            # Filter results based on the query
            filtered_df, error_message = filter_results(user_input, results)
            if error_message:
                st.warning(error_message)
            else:
                st.success(f"Results for your query:")
                st.table(filtered_df)

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Enter a query above to search for products.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("💡 Powered by **ChromaDB** and **Streamlit**")
