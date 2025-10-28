import streamlit as st
import pickle

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit App Title
st.title("📧 Email Spam Detection App")
st.write("This app predicts whether an email message is **Spam** or **Not Spam** using Machine Learning.")

# Input area
input_mail = st.text_area("Enter the email text:")

# Predict button
if st.button("Predict"):
    if input_mail.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        # Preprocess and predict
        input_data = vectorizer.transform([input_mail])
        prediction = model.predict(input_data)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This is a SPAM message!")
        else:
            st.success("✅ This is NOT a spam message.")


# Footer / Credits
st.markdown("---")
st.markdown("*Prepared by:* Om Jena, Vinay Surana, and Biswojit Mohapatra")
