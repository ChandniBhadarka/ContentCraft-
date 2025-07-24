import streamlit as st
from google import genai
from google.genai import types
import os
from streamlit_carousel import carousel
import base64
import requests
from io import BytesIO
import urllib.parse

# Try to import API keys
try:
    from apikey import google_gemini_api_key
except ImportError:
    st.error("❌ Could not import API keys. Please make sure 'apikey.py' file exists with your Google Gemini API key.")
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

def generate_images(blog_title, num_img):
    """Generate images using Unsplash API or create AI-generated image descriptions"""
    images_gallery = []
    
    if not google_gemini_api_key or google_gemini_api_key.strip() == "":
        st.error("Google Gemini API key is not configured properly.")
        return []
    
    try:
        # Initialize Gemini client for generating image concepts
        gemini_client = genai.Client(api_key=google_gemini_api_key)
        model = "gemini-2.0-flash-exp"
        
        for i in range(num_img):
            with st.spinner(f"Generating image {i+1} of {num_img}..."):
                
                # Generate relevant keywords for the image
                keyword_prompt = f"Based on the blog title '{blog_title}', suggest 2-3 relevant keywords that would be good for finding stock photos. Return only the keywords separated by commas, no other text."
                
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=keyword_prompt)
                        ],
                    ),
                ]
                
                generate_content_config = types.GenerateContentConfig(
                    response_mime_type="text/plain",
                )
                
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                
                if response and response.text:
                    # Extract keywords and use them to get relevant images
                    keywords = response.text.strip().replace(',', ' ').replace('\n', ' ')
                    
                    # Try to get image from Unsplash (free service)
                    try:
                        # Clean the keywords for URL
                        search_query = urllib.parse.quote(keywords[:50])  # Limit length
                        unsplash_url = f"https://source.unsplash.com/1200x800/?{search_query}"
                        
                        # Test if the URL is accessible
                        response_test = requests.get(unsplash_url, timeout=10)
                        if response_test.status_code == 200:
                            image_url = unsplash_url
                        else:
                            # Fallback to a themed placeholder
                            image_url = f"https://via.placeholder.com/1200x800/667eea/ffffff?text={urllib.parse.quote(f'Blog: {blog_title[:20]}...')}"
                            
                    except Exception:
                        # Fallback to placeholder
                        image_url = f"https://via.placeholder.com/1200x800/667eea/ffffff?text={urllib.parse.quote(f'Blog Image {i+1}')}"
                    
                    new_image = single_image.copy()
                    new_image["title"] = f"Image {i+1}: {blog_title}"
                    new_image["text"] = f"Generated for: {keywords}"
                    new_image["img"] = image_url
                    images_gallery.append(new_image)
                    
                else:
                    # Simple fallback
                    fallback_url = f"https://via.placeholder.com/1200x800/4a90e2/ffffff?text={urllib.parse.quote(f'Blog Image {i+1}')}"
                    new_image = single_image.copy()
                    new_image["title"] = f"Image {i+1}"
                    new_image["text"] = f"Image for: {blog_title}"
                    new_image["img"] = fallback_url
                    images_gallery.append(new_image)
            
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error generating images: {error_msg}")
        
        # Create fallback images even if there's an error
        for i in range(num_img):
            fallback_url = f"https://via.placeholder.com/1200x800/8b5cf6/ffffff?text={urllib.parse.quote(f'Blog Image {i+1}')}"
            new_image = single_image.copy()
            new_image["title"] = f"Image {i+1}"
            new_image["text"] = f"Placeholder for: {blog_title}"
            new_image["img"] = fallback_url
            images_gallery.append(new_image)
        
    return images_gallery

# Streamlit App Configuration
st.set_page_config(
    page_title="ContentCraft",
    page_icon="✍️",
    layout="wide"
)

# Check if API key is available
if not google_gemini_api_key:
    st.error("⚠️ Google Gemini API key not found! Please check your apikey.py file.")
    st.stop()

# Main title
st.title('✍️ ContentCraft: Your AI Writing Companion 🤖')

# Subheader
st.subheader("Craft your blog perfectly with the help of AI. ContentCraft is your new AI Blog Companion powered by Google Gemini. 🖋️📖")

# Sidebar for user input
with st.sidebar:
    st.title("Input for your blog content")
    st.subheader("Enter details of blog you want to generate 📝")

    # Blog title
    blog_title = st.text_input("Blog Title", placeholder="Enter your blog title here")
    
    # Keywords input
    keywords = st.text_area("Enter Keywords (comma-separated)", placeholder="AI, technology, innovation..")
    
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
                images = generate_images(blog_title, num_img)
                
                if images:
                    # Display images in a carousel
                    carousel(items=images, width=1.0)
                else:
                    st.warning("Could not generate images. Blog content will still be generated.")
            
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
st.markdown("**Note:** Make sure you have a valid Google Gemini API key configured in your `apikey.py` file with Imagen access enabled.")
