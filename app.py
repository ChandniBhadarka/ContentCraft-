import streamlit as st
from google import genai
from google.genai import types
import os
from streamlit_carousel import carousel
import base64
import requests
from io import BytesIO
import urllib.parse
import time

# For local OpenDalle model
try:
    from diffusers import AutoPipelineForText2Image
    import torch
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


try:
    from apikey import google_gemini_api_key, huggingface_api_key
except ImportError:
    st.error("❌ Could not import API keys. Please make sure 'apikey.py' file exists with your Google Gemini API key and Hugging Face API key.")
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
        
        model = "gemini-2.0-flash-exp" 
        
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
        
        
        response = gemini_client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        return response.text
        
    except Exception as e:
        st.error(f"Error generating blog content: {str(e)}")
        return None

def generate_image_prompt(blog_title, image_number):
    """Generate a detailed image prompt using Gemini API with fallback"""
    try:
       
        if not google_gemini_api_key or google_gemini_api_key.strip() == "":
            raise Exception("Google Gemini API key not found")
        
       
        gemini_client = genai.Client(api_key=google_gemini_api_key)
        model = "gemini-2.0-flash-exp"
        
        prompt_request = f"""Based on the blog title '{blog_title}', create a detailed, descriptive prompt for generating an image that would be suitable for this blog post. 
        The prompt should be specific, visually descriptive, and professional. 
        Make it suitable for AI image generation. Keep it under 200 characters.
        This is image {image_number} for the blog.
        
        Format: Return only the image generation prompt, no other text."""
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt_request)
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
            return response.text.strip()
        else:
           
            raise Exception("Empty response from Gemini")
            
    except Exception as e:
       
        print(f"Warning: Could not generate custom prompt with Gemini: {str(e)}")
        
       
        fallback_prompts = {
            1: f"Professional illustration related to {blog_title}, high quality, detailed, masterpiece",
            2: f"Modern graphic design about {blog_title}, clean, professional, vibrant colors",
            3: f"Creative visualization of {blog_title}, artistic, professional, eye-catching"
        }
        
       
        prompt_key = image_number if image_number in fallback_prompts else 1
        
        return fallback_prompts[prompt_key]

@st.cache_resource
def load_opendalle_model():
    """Load the OpenDalle model - cached to avoid reloading"""
    if not DIFFUSERS_AVAILABLE:
        return None
        
    try:
        model_id = "dataautogpt3/OpenDalleV1.1"
        
        with st.spinner("Loading OpenDalle model (this may take a few minutes on first run)..."):
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            
            
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                st.success("✅ OpenDalle model loaded on GPU")
            else:
                pipe = pipe.to("cpu")
                st.warning("⚠️ OpenDalle model loaded on CPU (slower generation)")
                
        return pipe
        
    except Exception as e:
        st.error(f"Failed to load OpenDalle model: {str(e)}")
        return None

def generate_images_with_opendalle_local(blog_title, num_img):
    """Generate images using local OpenDalle model"""
    images_gallery = []
    
    if not DIFFUSERS_AVAILABLE:
        st.error("❌ Required libraries not installed. Please install: pip install diffusers torch transformers")
        return []
    
    # Load the model
    pipe = load_opendalle_model()
    if pipe is None:
        st.error("❌ Failed to load OpenDalle model")
        return []
    
    try:
        for i in range(num_img):
            with st.spinner(f"Generating image {i+1} of {num_img} with OpenDalle..."):
                
                # Generate a detailed prompt for the image
                image_prompt = generate_image_prompt(blog_title, i+1)
                
                # Enhance the prompt for better results
                enhanced_prompt = f"{image_prompt}, best quality, extremely detailed, professional, high resolution, masterpiece"
                st.info(f"Generating with prompt: {enhanced_prompt[:100]}...")
                
                try:
                    # Generate the image
                    with torch.inference_mode():
                        image = pipe(
                            prompt=enhanced_prompt,
                            num_inference_steps=50,  # Good balance of quality and speed
                            guidance_scale=7.5,
                            width=512,
                            height=512
                        ).images[0]
                    
                    # Convert PIL image to base64 for display
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    image_data_url = f"data:image/png;base64,{img_base64}"
                    
                    new_image = single_image.copy()
                    new_image["title"] = f"OpenDalle Generated Image {i+1}"
                    new_image["text"] = f"Prompt: {image_prompt[:100]}..."
                    new_image["img"] = image_data_url
                    images_gallery.append(new_image)
                    
                    st.success(f"✅ Successfully generated image {i+1}")
                    
                except Exception as gen_e:
                    st.error(f"Error generating image {i+1}: {str(gen_e)}")
                    # Add placeholder
                    fallback_url = f"https://via.placeholder.com/512x512/8b5cf6/ffffff?text={urllib.parse.quote(f'Generation Error {i+1}')}"
                    new_image = single_image.copy()
                    new_image["title"] = f"Error Image {i+1}"
                    new_image["text"] = f"Error: {str(gen_e)[:50]}..."
                    new_image["img"] = fallback_url
                    images_gallery.append(new_image)
                
                # Clear GPU cache to prevent memory issues
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
    except Exception as e:
        st.error(f"Error in OpenDalle image generation: {str(e)}")
        
        # Create fallback images
        for i in range(num_img):
            fallback_url = f"https://via.placeholder.com/512x512/8b5cf6/ffffff?text={urllib.parse.quote(f'OpenDalle Error {i+1}')}"
            new_image = single_image.copy()
            new_image["title"] = f"Error Image {i+1}"
            new_image["text"] = f"OpenDalle generation failed"
            new_image["img"] = fallback_url
            images_gallery.append(new_image)
    
    return images_gallery

def generate_images_pollinations(blog_title, num_img):
    """Generate images using Pollinations.ai (free, no API key required)"""
    images_gallery = []
    
    # Simple prompt templates
    prompt_templates = [
        f"professional illustration {blog_title} high quality detailed",
        f"modern graphic design {blog_title} colorful professional",
        f"abstract art {blog_title} creative professional",
        f"minimalist design {blog_title} clean elegant",
        f"digital art {blog_title} vibrant detailed"
    ]
    
    try:
        for i in range(num_img):
            with st.spinner(f"Generating image {i+1} with Pollinations.ai..."):
                
                # Get prompt
                prompt_index = i % len(prompt_templates)
                prompt = prompt_templates[prompt_index]
                
                # st.info(f"🎨 Using prompt: {prompt}")
                
                # Create URL for Pollinations.ai
                encoded_prompt = urllib.parse.quote(prompt)
                seed = i + 1  # Different seed for each image
                image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&seed={seed}&model=flux"
                
                try:
                    # Fetch the image
                    response = requests.get(image_url, timeout=30)
                    if response.status_code == 200 and len(response.content) > 1000:
                        image_base64 = base64.b64encode(response.content).decode()
                        image_data_url = f"data:image/png;base64,{image_base64}"
                        
                        new_image = single_image.copy()
                        new_image["title"] = f"Generated Image {i+1}"
                        new_image["text"] = f"Prompt: {prompt}"
                        new_image["img"] = image_data_url
                        images_gallery.append(new_image)
                        
                        # st.success(f"✅ Generated image {i+1}")
                    else:
                        st.warning(f"⚠️ Failed to generate image {i+1} - Status: {response.status_code}")
                        # Add placeholder
                        fallback_url = f"https://via.placeholder.com/512x512/ff6b6b/ffffff?text={urllib.parse.quote(f'Failed {i+1}')}"
                        new_image = single_image.copy()
                        new_image["title"] = f"Failed Image {i+1}"
                        new_image["text"] = f"Generation failed: {prompt[:50]}..."
                        new_image["img"] = fallback_url
                        images_gallery.append(new_image)
                        
                except Exception as e:
                    st.warning(f"⚠️ Error generating image {i+1}: {str(e)}")
                    # Add placeholder
                    fallback_url = f"https://via.placeholder.com/512x512/ff6b6b/ffffff?text={urllib.parse.quote(f'Error {i+1}')}"
                    new_image = single_image.copy()
                    new_image["title"] = f"Error Image {i+1}"
                    new_image["text"] = f"Error: {str(e)[:50]}..."
                    new_image["img"] = fallback_url
                    images_gallery.append(new_image)
                
                time.sleep(1)  # Be nice to the free service
                
    except Exception as e:
        st.error(f"❌ Error with Pollinations API: {str(e)}")
        
        # Create fallback images
        for i in range(num_img):
            fallback_url = f"https://via.placeholder.com/512x512/ff6b6b/ffffff?text={urllib.parse.quote(f'API Error {i+1}')}"
            new_image = single_image.copy()
            new_image["title"] = f"Error Image {i+1}"
            new_image["text"] = f"Pollinations API failed"
            new_image["img"] = fallback_url
            images_gallery.append(new_image)
    
    return images_gallery

def generate_images_with_api_fallback(blog_title, num_img):
    """Fallback to API-based image generation if local model fails"""
    images_gallery = []
    
    if not huggingface_api_key or huggingface_api_key.strip() == "":
        st.error("Hugging Face API key is not configured properly.")
        return []
    
    # Use reliable API models as fallback
    model_options = [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1",
        "CompVis/stable-diffusion-v1-4"
    ]
    
    headers = {"Authorization": f"Bearer {huggingface_api_key}"}
    working_model = model_options[0]  # Default to first model
    API_URL = f"https://api-inference.huggingface.co/models/{working_model}"
    
    try:
        for i in range(num_img):
            with st.spinner(f"Generating fallback image {i+1} of {num_img}..."):
                
                # Generate a detailed prompt for the image
                image_prompt = generate_image_prompt(blog_title, i+1)
                enhanced_prompt = f"{image_prompt}, high quality, detailed, professional"
                
                payload = {"inputs": enhanced_prompt}
                
                # Try to generate with API
                max_retries = 2
                success = False
                
                for attempt in range(max_retries):
                    try:
                        response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
                        
                        if response.status_code == 200:
                            image_bytes = response.content
                            image_base64 = base64.b64encode(image_bytes).decode()
                            image_data_url = f"data:image/png;base64,{image_base64}"
                            
                            new_image = single_image.copy()
                            new_image["title"] = f"AI Generated Image {i+1}"
                            new_image["text"] = f"Prompt: {image_prompt[:100]}..."
                            new_image["img"] = image_data_url
                            images_gallery.append(new_image)
                            
                            st.success(f"✅ Generated fallback image {i+1}")
                            success = True
                            break
                            
                        elif response.status_code == 503:
                            if attempt < max_retries - 1:
                                st.warning(f"Model loading, retrying in 10 seconds...")
                                time.sleep(10)
                                continue
                            
                    except Exception as req_e:
                        if attempt < max_retries - 1:
                            time.sleep(5)
                            continue
                
                # Add placeholder if all attempts failed
                if not success:
                    fallback_url = f"https://via.placeholder.com/512x512/8b5cf6/ffffff?text={urllib.parse.quote(f'API Image {i+1}')}"
                    new_image = single_image.copy()
                    new_image["title"] = f"Placeholder Image {i+1}"
                    new_image["text"] = f"Could not generate: {image_prompt[:50]}..."
                    new_image["img"] = fallback_url
                    images_gallery.append(new_image)
                
                time.sleep(2)  # Rate limiting
                
    except Exception as e:
        st.error(f"API fallback error: {str(e)}")
        
    return images_gallery

# Streamlit App Configuration
st.set_page_config(
    page_title="ContentCraft",
    page_icon="✍️",
    layout="wide"
)

# Check if API keys are available
if not google_gemini_api_key:
    st.error("⚠️ Google Gemini API key not found! Please check your apikey.py file.")
    st.stop()

# Main title
st.title('✍️ ContentCraft: Your AI Writing Companion 🤖')

# Subheader
st.subheader("Craft your blog perfectly with the help of AI. ContentCraft is your new AI Blog Companion powered by Google Gemini and Pollinations.ai. 🖋️📖")

# Show system info
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.info(f"🔧 Diffusers Available: {'✅' if DIFFUSERS_AVAILABLE else '❌'}")
# with col2:
#     if DIFFUSERS_AVAILABLE and torch.cuda.is_available():
#         st.info("🚀 GPU Available: ✅")
#     else:
#         st.info("💻 Running on: CPU")
# with col3:
#     st.info("🎨 Pollinations.ai: ✅ (Free)")

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
    
    # Image generation method
    if DIFFUSERS_AVAILABLE:
        image_method = st.radio(
            "Image Generation Method",
            ["Pollinations.ai (Free)"],
            help="Pollinations.ai is free and reliable. OpenDalle (Local) provides better quality but requires GPU."
        )
    else:
        image_method = st.radio(
            "Image Generation Method",
            ["Pollinations.ai (Free)", "Hugging Face API"],
            help="Pollinations.ai is free and doesn't require API keys."
        )
    
    # Number of images
    num_img = st.number_input("Number of images", min_value=0, max_value=3, step=1, value=1)

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
                st.subheader("🖼️ AI Generated Images:")
                
                if image_method == "Pollinations.ai (Free)":
                    # st.info("🌐 Using Pollinations.ai (Free)")
                    images = generate_images_pollinations(blog_title, num_img)
                    
                # elif image_method == "OpenDalle (Local)" and DIFFUSERS_AVAILABLE:
                #     st.info("🎨 Using OpenDalle V1.1 (Local Model)")
                #     images = generate_images_with_opendalle_local(blog_title, num_img)
                    
                # elif image_method == "Hugging Face API":
                #     st.info("🤗 Using Hugging Face API")
                #     images = generate_images_with_api_fallback(blog_title, num_img)
                    
                # elif image_method == "Both":
                #     st.info("🎯 Using both Pollinations.ai and OpenDalle")
                #     # Use Pollinations for half, OpenDalle for the rest
                #     half = max(1, num_img // 2)
                #     images = generate_images_pollinations(blog_title, half)
                #     if num_img > half and DIFFUSERS_AVAILABLE:
                #         additional_images = generate_images_with_opendalle_local(blog_title, num_img - half)
                #         images.extend(additional_images)
                #     elif num_img > half:
                #         additional_images = generate_images_pollinations(blog_title, num_img - half)
                #         images.extend(additional_images)
                
                else:
                    # st.info("🌐 Using Pollinations.ai (default)")
                    images = generate_images_pollinations(blog_title, num_img)
                
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
                # st.download_button(
                #     label="Download Blog Content",
                #     data=blog_content,
                #     file_name=f"{blog_title.replace(' ', '_')}_blog.txt",
                #     mime="text/plain"
                # )
            else:
                st.error("Could not generate blog content. Please check your Gemini API key and try again.")

