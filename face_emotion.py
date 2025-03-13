import streamlit as st
from llama_cpp import Llama
from PIL import Image
import numpy as np
from llama_cpp.llama_chat_format import Llava15ChatHandler
import os

MODEL_PATH = "/home/usha/Desktop/work/llama.cpp/models/ggml-model-q4_k.gguf"
IMAGE_SIZE = 224  # LLaVA's recommended image size

def process_image(uploaded_image):
    # Load and resize image
    image = Image.open(uploaded_image).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(image, dtype=np.uint8)

def detect_emotion(image_array):
    # Initialize model and projector
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=15,
        n_ctx=1024,  # Increase context window for better comprehension
        logits_all=True,
        chat_handler=Llava15ChatHandler(clip_model_path="mmproj-model-f16.gguf"),
        verbose=False
    )
    
    # Create system prompt
    system_prompt = """You are a strict facial emotion classification system. 
Analyze the facial expression and respond ONLY with ONE word from this list:
**happy, sad, angry, surprised, fearful, disgusted, neutral**
Absolutely no explanations or punctuation."""
    # Create user prompt
    user_prompt = """Return only the emotion name in lowercase without any formatting.
The emotion must be one of: happy, sad, angry, surprised, fearful, disgusted, neutral"""

    # Process the image and generate response
    response = llm.create_chat_completion(
        messages=[{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": [{"type": "image", "data": image_array}, user_prompt]
        }],
        temperature=0.1,  # Low temperature for deterministic responses
        max_tokens=20
    )
    return response['choices'][0]['message']['content']

def main():
    st.title("Facial Emotion Detection")
    st.write("Upload a facial image for emotion analysis")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at {MODEL_PATH}")
            return

        # Display image
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Process image and detect emotion
        with st.spinner("Analyzing emotion..."):
            try:
                image_array = process_image(uploaded_file)
                emotion = detect_emotion(image_array)
                
                # Post-process response to get clean emotion
                emotion2 = postprocess_emotion(emotion)
                
                with col2:
                    st.subheader("Detection Result")
                    st.markdown(f"**Emotion:** {emotion2.capitalize()}")
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

def postprocess_emotion(response):
    emoji_map = {
        '😊': 'happy',
        '😄': 'happy',
        '😢': 'sad',
        '😠': 'angry',
        '😲': 'surprised',
        '😨': 'fearful',
        '🤢': 'disgusted',
        '😐': 'neutral'
    }
    
    emotion = response.strip().lower()
    emotions_list = ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"]
    
    # Check for emojis
    for emoji, label in emoji_map.items():
        if emoji in emotion:
            return label
    
    # Check for partial matches
    for e in emotions_list:
        if e in emotion:
            return e
    
    return "neutral"

if __name__ == "__main__":
    os.environ["LLAMA_CUBLAS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()

