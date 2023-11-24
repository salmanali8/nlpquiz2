import streamlit as st

# Attempt to import transformers
try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    transformers_available = False

# Streamlit app
def main():
    st.title("Text Classification App")
    
    if not transformers_available:
        st.error("Please install the transformers library to use this app.")
        st.write("You can install it via terminal using:")
        st.code("pip install transformers")
        st.stop()

    classifier = pipeline("sentiment-analysis")

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
