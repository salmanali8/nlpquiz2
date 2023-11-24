import streamlit as st
from transformers import pipeline

# Function to make predictions
def classify_text(text, model_name):
    classifier = pipeline("text-classification", model=model_name)
    prediction = classifier(text)
    return prediction[0]

def main():
    st.title("Text Classification App")
    
    model_name = st.text_input("Enter the Hugging Face model name (e.g., distilbert-base-uncased)")
    text = st.text_area("Enter text for classification")

    if st.button("Classify"):
        if model_name.strip() == "":
            st.warning("Please enter a valid Hugging Face model name.")
        elif text.strip() == "":
            st.warning("Please enter some text for classification.")
        else:
            with st.spinner("Classifying..."):
                prediction = classify_text(text, model_name)
                st.write(f"Predicted label: {prediction['label']}")
                st.write(f"Confidence score: {prediction['score']}")

if __name__ == "__main__":
    main()
