# Facial-Emotion-Detection-with-Jetson-Orin-Nano

Inferring emotions from multiple modalities is very critical for social communication and deficits in emotion recognition are a very important marker in the diagnosis of autism spectrum disorder. This project uses Llama 3.1 and Jetson Nano orin kit to help autistic individuals recognize emotions. 



This is a lightweight facial emotion detection system using LLaVA (Large Language-and-Vision Assistant). I have optimized it to run on devices like the NVIDIA Jetson Orin Nano with 8GB RAM.
## Demo


https://github.com/user-attachments/assets/caede134-31ec-4336-beae-41a2b475d328


## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/facial-emotion-detection.git
   cd facial-emotion-detection

2. LLaVa model:
   ```bash
   wget https://huggingface.co/mys/ggml_llava-v1.6-vicuna-3b/resolve/main/ggml-model-q4_k.gguf -O models/ggml-model-q4_k.gguf

3. Clip Projector:
     ```bash
     wget https://huggingface.co/mys/ggml_llava-v1.6-vicuna-3b/resolve/main/mmproj-model-f16.gguf -O models/mmproj-model-f16.gguf

5. Update the model path in the code and set the gpu threads and context size as per the memory of the system
## Usage 
#### Run the Application:
```bash
streamlit run face_emotion.py
```

