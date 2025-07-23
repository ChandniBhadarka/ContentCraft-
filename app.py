import streamlit as st
from google import genai
from google.genai import types
import os
from openai import OpenAI
from streamlit_carousel import carousel

# Try to import API keys
try:
    from apikey import google_gemini_api_key, openai_api_key
except ImportError:
    st.error("❌ Could not import API keys. Please make sure 'apikey.py' file exists with your API keys.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error importing API keys: {str(e)}")
    st.stop()

single_image = dict(
    title="",
    text="",
    img="",
    link="https://discuss.streamlit.io/t/new-component-react-bootstrap-carousel/46819",
)

def generate_blog_content(blog_title, keywords, num_words):
    """Generate blog content using Google Gemini API"""
    try:
        # Configure the Gemini client
        gemini_client = genai.Client(api_key=google_gemini_api_key)
        
        model = "gemini-2.0-flash-exp"  # Updated to a current model
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"""Generate a comprehensive blog post relevant to the title "{blog_title}" and keywords "{keywords}". 
                        Make sure to incorporate these keywords naturally in the blog post. 
                        The blog should be approximately {num_words} words in length, suitable for an online audience. 
                        Ensure the content is original, informative, and maintain a consistent tone throughout.
                        Please provide only the blog content without any meta-commentary or analysis."""
                    ),
                ],
            ),
        ]
        
        tools = [
            types.Tool(googleSearch=types.GoogleSearch()),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            tools=tools,
            response_mime_type="text/plain",
        )
        
        # Generate content
        response = gemini_client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        return response.text
        
    except Exception as e:
        st.error(f"Error generating blog content: {str(e)}")
        return None

def test_openai_connection():
    """Test OpenAI API connection"""
    try:
        client = OpenAI(
            api_key=openai_api_key,
            timeout=30.0,
            max_retries=1
        )
        # Try a simple API call to test connection
        models = client.models.list()
        return True, "OpenAI connection successful"
    except Exception as e:
        return False, str(e)

def generate_images(blog_title, num_img):
    """Generate images using OpenAI DALL-E"""
    images_gallery = []
    
    if not openai_api_key or openai_api_key.strip() == "":
        st.error("OpenAI API key is not configured properly.")
        return []
    
    try:
        # Initialize OpenAI client with explicit configuration
        client = OpenAI(
            api_key=openai_api_key,
            timeout=60.0,  # Set timeout
            max_retries=2   # Set max retries
        )
        
        for i in range(num_img):
            with st.spinner(f"Generating image {i+1} of {num_img}..."):
                image_response = client.images.generate(
                    model="dall-e-3",
                    prompt=f"Generate a professional blog post image related to: {blog_title}. Make it visually appealing and relevant to the topic.",
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                
                new_image = single_image.copy()
                new_image["title"] = f"Image {i+1}"
                new_image["text"] = f"{blog_title}"
                new_image["img"] = image_response.data[0].url
                images_gallery.append(new_image)
            
    except Exception as e:
        error_msg = str(e)
        if "proxies" in error_msg:
            st.error("Network configuration error. Try updating the OpenAI library: `pip install --upgrade openai`")
        elif "api_key" in error_msg.lower():
            st.error("Invalid OpenAI API key. Please check your API key configuration.")
        elif "quota" in error_msg.lower():
            st.error("OpenAI API quota exceeded. Please check your usage limits.")
        else:
            st.error(f"Error generating images: {error_msg}")
        return []
        
    return images_gallery

# Streamlit App Configuration
st.set_page_config(
    page_title="ContentCraft",
    page_icon="✍️",
    layout="wide"
)

# Check if API keys are available
if not google_gemini_api_key or not openai_api_key:
    st.error("⚠️ API keys not found! Please check your apikey.py file.")
    st.stop()

# Main title
st.title('✍️ ContentCraft: Your AI Writing Companion 🤖')

# Subheader
st.subheader("Craft your blog perfectly with the help of AI. ContentCraft is your new AI Blog Companion. 🖋️📖")

# Sidebar for user input
with st.sidebar:
    st.title("Input for your blog content")
    st.subheader("Enter details of blog you want to generate 📝")

    # Blog title
    blog_title = st.text_input("Blog Title", placeholder="Enter your blog title here...")
    
    # Keywords input
    keywords = st.text_area("Enter Keywords (comma-separated)", placeholder="AI, technology, innovation...")
    
    # Number of words
    num_words = st.slider("Number of words", min_value=200, max_value=2000, step=100, value=800)
    
    # Number of images
    num_img = st.number_input("Number of images", min_value=0, max_value=5, step=1, value=1)

    # Submit button
    submit_button = st.button("Generate Blog", type="primary")

# Main content area
if submit_button:
    if not blog_title.strip():
        st.error("Please enter a blog title!")
    elif not keywords.strip():
        st.error("Please enter at least one keyword!")
    else:
        # Show progress
        with st.spinner("Generating your blog content and images..."):
            
            # Generate images if requested
            if num_img > 0:
                st.subheader("🖼️ Generated Images:")
                
                # Test OpenAI connection first
                # connection_ok, connection_msg = test_openai_connection()
                # if not connection_ok:
                #     st.error(f"OpenAI connection failed: {connection_msg}")
                #     if "proxies" in connection_msg:
                #         st.info("💡 **Solution**: Try running: `pip install --upgrade openai`")
                #     elif "authentication" in connection_msg.lower():
                #         st.info("💡 **Solution**: Check your OpenAI API key in apikey.py")
                #     st.warning("Skipping image generation due to connection issues.")
                # else:
                #     images_gallery = generate_images(blog_title, num_img)
                    
                #     if images_gallery:
                #         carousel(items=images_gallery, width=1)
                #     else:
                #         st.warning("Could not generate images. Please try again.")
            
            # Generate blog content
            st.subheader("📝 Your Generated Blog:")
            blog_content = generate_blog_content(blog_title, keywords, num_words)
            
            if blog_content:
                st.markdown(blog_content)
                
                # Add download button for the blog content
                st.download_button(
                    label="Download Blog Content",
                    data=blog_content,
                    file_name=f"{blog_title.replace(' ', '_')}_blog.txt",
                    mime="text/plain"
                )
            else:
                st.error("Could not generate blog content. Please check your Gemini API key and try again.")

# Footer
st.markdown("---")
st.markdown("**Note:** Make sure you have valid API keys for both OpenAI and Google Gemini configured in your `apikey.py` file.")
