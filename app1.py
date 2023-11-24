import streamlit as st
from transformers import pipeline

# Load pre-trained model for text classification
classifier = pipeline("sentiment-analysis")

# Streamlit app
def main():
    st.title("Text Classification App")

    # User input for text
    text_input = st.text_area("Enter text for classification:")

    if st.button("Classify"):
        if text_input:
            # Perform classification
            result = classifier(text_input)
            
            # Display result
            st.write("### Classification Result:")
            for res in result:
                st.write(f"Label: {res['label']}, Confidence: {res['score']}")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
