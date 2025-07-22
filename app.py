import streamlit as st


st.set_page_config(layout="wide")
#title
st.title('✍️ContentCraft : Your AI Writing Companion 🤖')

#Subheader
st.subheader("You can craft your blog perfectly with the help of AI. ContentCraft is your new AI Blog Companion.🖋️📖")

# sidebar for user input
with st.sidebar:
    st.title("Input for your blog content")
    st.subheader("Enter details of blog you want to generate 📝")

    #blog title
    blog_title = st.text_input("Blog Title")
    # keyword input
    keywords = st.text_area("Enter Keywords(comma-separated)")
    #number of words
    num_words = st.slider("Number of words", min_value=200, max_value=2000, step=100)
    #number of images
    num_img = st.number_input("Number of images", min_value=0, max_value=5, step=1)

    #submit button
    submit_button = st.button("Generate Blog")