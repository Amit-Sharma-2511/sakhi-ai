import gradio as gr
import torch
import os
import pyttsx3
import tempfile
import speech_recognition as sr
import time
import uuid
from tiny_gpt import GPTLanguageModel, get_sakhi_response, device, model_path, decode, encode

# Initialize model
model = GPTLanguageModel().to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

class SakhiState:
    def __init__(self):
        self.chat_tokens = []

def get_audio_response(text):
    try:
        # Robust initialization for Windows
        engine = pyttsx3.init('sapi5')
        voices = engine.getProperty('voices')
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)
        engine.setProperty('rate', 160)
        
        temp_dir = tempfile.gettempdir()
        unique_filename = f"sakhi_voice_{uuid.uuid4().hex}.wav"
        file_path = os.path.join(temp_dir, unique_filename)
        
        engine.save_to_file(text, file_path)
        engine.runAndWait()
        
        time.sleep(0.1) # Release handle
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return file_path
        return None
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def chat_function(message, history, state, voice_mode):
    if not message: return history, None, state
    if state is None: state = SakhiState()
    if history is None: history = []
    
    response, new_tokens = get_sakhi_response(message, state.chat_tokens, model)
    state.chat_tokens = new_tokens
    
    audio_path = get_audio_response(response) if voice_mode else None
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    
    return history, audio_path, state

def voice_chat(audio_path, history, state, voice_enabled):
    if audio_path is None: return history, None, state
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)
        user_text = r.recognize_google(audio, language='en-in')
        return chat_function(user_text, history, state, voice_enabled)
    except:
        return history, None, state

def clear_session():
    return [], None, SakhiState(), ""

# --- PREMIUM GLASSMORPHISM NIGHT UI ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=Playfair+Display:wght@700&display=swap');

:root {
    --bg-silk: #0f1014;
    --glass-bg: rgba(255, 255, 255, 0.03);
    --glass-border: rgba(255, 255, 255, 0.08);
    --primary-electric: #8b5cf6; /* Electric Violet */
    --accent-gold: #fbbf24;
    --text-pure: #ffffff;
    --text-dim: #94a3b8;
}

body { 
    background-color: var(--bg-silk);
    background-image: radial-gradient(circle at 50% -20%, #1e1b4b 0%, var(--bg-silk) 80%);
    font-family: 'Outfit', sans-serif;
    color: var(--text-pure);
    margin: 0;
    padding: 0;
}

.gradio-container { background: transparent !important; }

/* Centered Layout */
.app-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 20px;
    display: flex;
    flex-direction: column;
    gap: 30px;
}

/* Glass Card */
.glass-panel {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border-radius: 28px;
    border: 1px solid var(--glass-border);
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    padding: 30px;
}

/* Header Styling */
.header-unit {
    text-align: center;
    margin-bottom: 20px;
}
.header-unit h1 {
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.header-unit p {
    color: var(--text-dim);
    font-size: 16px;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 300;
}

/* Chatbot Overhaul */
.chatbot-premium {
    background: transparent !important;
    border: none !important;
}
.chatbot-premium .message.user {
    background: linear-gradient(135deg, var(--primary-electric) 0%, #6d28d9 100%) !important;
    border: none !important;
    border-radius: 20px 20px 0 20px !important;
    box-shadow: 0 10px 15px -3px rgba(139, 92, 246, 0.3);
}
.chatbot-premium .message.bot {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 20px 20px 20px 0 !important;
}

/* Floating Input Bar */
.input-hub {
    position: relative;
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 20px !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
.input-hub:focus-within {
    border-color: var(--primary-electric) !important;
    background: rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 0 30px rgba(139, 92, 246, 0.2);
}

/* Action Tags */
.btn-premium {
    border-radius: 14px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease;
}
.btn-electric { background: var(--primary-electric) !important; border: none !important; }
.btn-electric:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(139, 92, 246, 0.4); }

.control-pill {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 15px;
}
"""

with gr.Blocks(title="Sakhi AI") as demo:
    state = gr.State()
    
    with gr.Column(elem_classes="app-container"):
        # Elegance Header
        with gr.Column(elem_classes="header-unit"):
            gr.HTML("<h1>Sakhi</h1><p>Neural Interaction Hub</p>")
        
        # Main Glass Module
        with gr.Column(elem_classes="glass-panel"):
            chatbot = gr.Chatbot(
                label="", 
                height=550, 
                show_label=False, 
                elem_classes="chatbot-premium"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Command Sakhi...", 
                    show_label=False, 
                    container=False,
                    scale=10,
                    elem_classes="input-hub"
                )
                submit_btn = gr.Button("↑", variant="primary", scale=1, elem_classes="btn-premium btn-electric")

            with gr.Row(variant="compact"):
                audio_output = gr.Audio(label="Neural Playback", autoplay=True, visible=False)
                
            with gr.Row():
                with gr.Column(scale=1, elem_classes="control-pill"):
                    gr.Markdown("#### 🎙️ NEURAL MIC")
                    audio_input = gr.Audio(sources=["microphone"], type="filepath", show_label=False)
                with gr.Column(scale=1, elem_classes="control-pill"):
                    gr.Markdown("#### ⚙️ CORE OPS")
                    voice_mode = gr.Checkbox(label="Enable Neural Voice", value=True)
                    clear_btn = gr.Button("🗑️ PURGE STATE", variant="stop", elem_classes="btn-premium")

        gr.HTML("<div style='text-align:center; color:#444; font-size:12px; letter-spacing:2px;'>EXPERIMENTAL AGENT VER 5.2.0</div>")

    # Mappings
    msg.submit(chat_function, [msg, chatbot, state, voice_mode], [chatbot, audio_output, state])
    msg.submit(lambda: "", None, msg)
    
    submit_btn.click(chat_function, [msg, chatbot, state, voice_mode], [chatbot, audio_output, state])
    submit_btn.click(lambda: "", None, msg)
    
    audio_input.change(voice_chat, [audio_input, chatbot, state, voice_mode], [chatbot, audio_output, state])
    
    clear_btn.click(clear_session, None, [chatbot, audio_output, state, msg])

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=custom_css)
