<script setup>
import { ref, onMounted, nextTick } from 'vue'
import axios from 'axios'

const messages = ref([
  { role: 'assistant', content: 'Namaste! I am Sakhi. Your premium Indian AI assistant. How can I help you today?' }
])
const inputMessage = ref('')
const isLoading = ref(false)
const isListening = ref(false)
const voiceMode = ref(true)
const chatTokens = ref([])
const scrollContainer = ref(null)

// Initialize Speech Recognition
let recognition = null
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
  recognition = new SpeechRecognition()
  recognition.continuous = false
  recognition.interimResults = false
  recognition.lang = 'en-IN'

  recognition.onresult = (event) => {
    const text = event.results[0][0].transcript
    inputMessage.value = text
    isListening.value = false
    sendMessage()
  }

  recognition.onend = () => {
    isListening.value = false
  }

  recognition.onerror = () => {
    isListening.value = false
  }
}

const toggleListening = () => {
  if (isListening.value) {
    recognition.stop()
  } else {
    isListening.value = true
    recognition.start()
  }
}

const scrollToBottom = async () => {
  await nextTick()
  if (scrollContainer.value) {
    scrollContainer.value.scrollTop = scrollContainer.value.scrollHeight
  }
}

const sendMessage = async () => {
  if (!inputMessage.value.trim() || isLoading.value) return

  const userMsg = inputMessage.value
  messages.value.push({ role: 'user', content: userMsg })
  inputMessage.value = ''
  isLoading.value = true
  
  await scrollToBottom()

  try {
    const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000'
    const response = await axios.post(`${apiBaseUrl}/chat`, {
      message: userMsg,
      tokens: chatTokens.value
    })

    const data = response.data
    messages.value.push({ role: 'assistant', content: data.response })
    chatTokens.value = data.tokens

    // Play Voice if enabled
    if (voiceMode.value && data.audio_url) {
      const audio = new Audio(`${apiBaseUrl}${data.audio_url}`)
      audio.play().catch(e => console.error("Audio playback failed:", e))
    }
  } catch (error) {
    console.error("Chat Error:", error)
    messages.value.push({ role: 'assistant', content: "I'm sorry, I'm having trouble connecting to my neural center. Please make sure the API is running." })
  } finally {
    isLoading.value = false
    await scrollToBottom()
  }
}

const clearChat = () => {
  messages.value = [{ role: 'assistant', content: 'Conversation cleared. How can I help you now?' }]
  chatTokens.value = []
}
</script>

<template>
  <div class="app-container">
    <!-- Glass Sidebar -->
    <aside class="sidebar">
      <div class="logo-area">
        <h1>SAKHI <span class="ai-tag">AI</span></h1>
        <p class="subtitle">Premium Neural Assistant</p>
      </div>
      
      <div class="controls">
        <div class="control-item">
          <span>Voice Output</span>
          <label class="switch">
            <input type="checkbox" v-model="voiceMode">
            <span class="slider round"></span>
          </label>
        </div>
      </div>

      <nav class="nav-links">
        <div class="nav-btn active">💬 Chat</div>
        <div class="nav-btn" @click="clearChat">🗑️ Clear Session</div>
      </nav>

      <div class="sidebar-footer">
        <div class="status-indicator">
          <span class="dot pulse"></span>
          <span>Core Online</span>
        </div>
        <p class="ver">Vtesting Responsive</p>
      </div>
    </aside>

    <!-- Main Chat Workspace -->
    <main class="chat-workspace">
      <header class="workspace-header">
        <div class="header-info">
          <h2>Neural Interaction Hub</h2>
          <p>Powered by Sakhi GPT v1.0</p>
        </div>
      </header>

      <div class="chat-viewport" ref="scrollContainer">
        <div v-for="(msg, index) in messages" :key="index" 
             :class="['bubble-wrap', msg.role === 'user' ? 'user-wrap' : 'bot-wrap']">
          <div class="bubble">
            <div class="msg-content">{{ msg.content }}</div>
          </div>
        </div>
        <div v-if="isLoading" class="bubble-wrap bot-wrap">
          <div class="bubble typing-bubble">
            <span class="dot-typing"></span>
            <span class="dot-typing"></span>
            <span class="dot-typing"></span>
          </div>
        </div>
      </div>

      <div class="input-workspace">
        <div class="input-hub glass">
          <button class="icon-btn" title="Add File">+</button>
          <input 
            v-model="inputMessage" 
            @keyup.enter="sendMessage"
            type="text" 
            placeholder="Talk to Sakhi..."
            :disabled="isLoading"
          />
          <div class="action-spread">
            <button @click="toggleListening" :class="['icon-btn mic-btn', { active: isListening }]" title="Voice Command">
              {{ isListening ? '⭕' : '🎙️' }}
            </button>
            <button @click="sendMessage" class="send-btn" :disabled="isLoading || !inputMessage.trim()">
              🚀
            </button>
          </div>
        </div>
        <p class="disclaimer">Sakhi AI is in Experimental Training Phase.</p>
      </div>
    </main>
  </div>
</template>

<style>
/* Global Styles */
:root {
  --primary: #6366f1;
  --primary-glow: rgba(99, 102, 241, 0.4);
  --bg-deep: #020617;
  --glass: rgba(255, 255, 255, 0.03);
  --border: rgba(255, 255, 255, 0.08);
}

body {
  margin: 0;
  background-color: var(--bg-deep);
  color: white;
  font-family: 'Outfit', sans-serif;
  overflow: hidden;
}

.app-container {
  display: flex;
  height: 100vh;
  background: radial-gradient(circle at top right, #1e1b4b 0%, #020617 100%);
}

/* Sidebar */
.sidebar {
  width: 280px;
  background: rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(20px);
  border-right: 1px solid var(--border);
  padding: 32px;
  display: flex;
  flex-direction: column;
  z-index: 10;
}

.logo-area h1 {
  font-size: 28px;
  font-weight: 800;
  letter-spacing: -1px;
  margin: 0;
}

.ai-tag {
  color: var(--primary);
  font-size: 14px;
  vertical-align: top;
  margin-left: 2px;
}

.subtitle {
  color: #64748b;
  font-size: 12px;
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 2px;
}

.controls {
  margin-top: 48px;
  background: var(--glass);
  padding: 20px;
  border-radius: 16px;
  border: 1px solid var(--border);
}

.control-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
  color: #94a3b8;
}

.nav-links {
  margin-top: 32px;
}

.nav-btn {
  padding: 14px 20px;
  border-radius: 12px;
  cursor: pointer;
  margin-bottom: 8px;
  transition: all 0.3s;
  color: #94a3b8;
  font-weight: 500;
}

.nav-btn:hover {
  background: var(--glass);
  color: white;
}

.nav-btn.active {
  background: rgba(99, 102, 241, 0.1);
  color: var(--primary);
}

.sidebar-footer {
  margin-top: auto;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #22c55e;
}

.dot {
  width: 8px;
  height: 8px;
  background: #22c55e;
  border-radius: 50%;
}

.pulse {
  box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  70% { box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); }
  100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
}

/* Chat Workspace */
.chat-workspace {
  flex: 1;
  display: flex;
  flex-direction: column;
  position: relative;
}

.workspace-header {
  padding: 32px 48px;
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(10px);
}

.header-info h2 {
  font-size: 18px;
  margin: 0;
}

.header-info p {
  font-size: 12px;
  color: #64748b;
  margin: 4px 0 0;
}

.chat-viewport {
  flex: 1;
  overflow-y: auto;
  padding: 40px;
  display: flex;
  flex-direction: column;
  gap: 32px;
}

/* Bubbles */
.bubble-wrap {
  display: flex;
  width: 100%;
}

.user-wrap { justify-content: flex-end; }
.bot-wrap { justify-content: flex-start; }

.bubble {
  max-width: 70%;
  padding: 16px 24px;
  border-radius: 24px;
  font-size: 16px;
  line-height: 1.6;
  position: relative;
  transition: transform 0.2s;
}

.user-wrap .bubble {
  background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
  color: white;
  border-bottom-right-radius: 4px;
  box-shadow: 0 10px 20px var(--primary-glow);
}

.bot-wrap .bubble {
  background: var(--glass);
  border: 1px solid var(--border);
  border-bottom-left-radius: 4px;
}

/* Input Area */
.input-workspace {
  padding: 20px 60px 40px;
}

.input-hub {
  max-width: 900px;
  margin: 0 auto;
  padding: 12px 24px;
  border-radius : 24px;
  display: flex;
  align-items: center;
  gap: 16px;
  border: 1px solid var(--border);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.input-hub.glass {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(30px);
}

.input-hub:focus-within {
  border-color: var(--primary);
  background: rgba(255, 255, 255, 0.08);
  box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}

.input-hub input {
  flex: 1;
  background: transparent;
  border: none;
  color: white;
  font-size: 16px;
  outline: none;
  padding: 12px 0;
}

.action-spread {
  display: flex;
  gap: 12px;
}

.icon-btn {
  background: none;
  border: none;
  color: #64748b;
  cursor: pointer;
  font-size: 18px;
  padding: 10px;
  border-radius: 12px;
  transition: all 0.2s;
}

.icon-btn:hover {
  background: rgba(255, 255, 255, 0.1);
  color: white;
}

.mic-btn.active {
  background: #ef4444;
  color: white;
  animation: mic-pulse 1.5s infinite;
}

@keyframes mic-pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.1); }
  100% { transform: scale(1); }
}

.send-btn {
  background: var(--primary);
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 14px;
  cursor: pointer;
  font-weight: 700;
  transition: all 0.2s;
}

.send-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px var(--primary-glow);
}

.send-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.disclaimer {
  text-align: center;
  font-size: 11px;
  color: #475569;
  margin-top: 16px;
}

/* Typing Animation */
.dot-typing {
  height: 6px;
  width: 6px;
  background: #94a3b8;
  border-radius: 50%;
  display: inline-block;
  margin-right: 4px;
  animation: typing 1s infinite alternate;
}

.dot-typing:nth-child(2) { animation-delay: 0.2s; }
.dot-typing:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
  from { transform: translateY(0); opacity: 0.4; }
  to { transform: translateY(-6px); opacity: 1; }
}

/* Switch Styles */
.switch {
  position: relative;
  display: inline-block;
  width: 36px;
  height: 20px;
}

.switch input { opacity: 0; width: 0; height: 0; }

.slider {
  position: absolute;
  cursor: pointer;
  top: 0; left: 0; right: 0; bottom: 0;
  background-color: #334155;
  transition: .4s;
}

.slider:before {
  position: absolute;
  content: "";
  height: 14px; width: 14px;
  left: 3px; bottom: 3px;
  background-color: white;
  transition: .4s;
}

input:checked + .slider { background-color: var(--primary); }
input:checked + .slider:before { transform: translateX(16px); }
.slider.round { border-radius: 34px; }
.slider.round:before { border-radius: 50%; }

/* Mobile Responsiveness */
@media (max-width: 900px) {
  .sidebar {
    position: absolute;
    left: -280px;
  }
  .input-workspace { padding: 20px; }
  .workspace-header { padding: 20px; }
  .chat-viewport { padding: 20px; }
  .bubble { max-width: 85%; }
}
</style>
