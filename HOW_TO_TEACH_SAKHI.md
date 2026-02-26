# 🎓 How to Teach Sakhi New Things

Sakhi is designed to learn from every interaction. Here are three ways you can make her smarter and teach her specific knowledge:

### 1. 🗨️ Learning Through Conversation (Online Learning)
Every time you chat with Sakhi via the dashboard, she does two things:
*   **Logs the Exchange**: She appends your conversation to `indian_chat.txt`.
*   **Instant Fine-tuning**: She immediately runs a few training steps on the specific words you just exchanged. This helps her "remember" the context of your current session.

### 2. 📝 Manual Teaching & Fine-Tuning
If you want Sakhi to know specific facts, follow these steps:
1.  Open the file **`indian_chat.txt`**.
2.  Add new lines in this exact format:
    ```text
    User: [Your Question or Statement]
    Sakhi: [The exact answer you want her to give]
    ```
3.  Save the file.
4.  Run the **Retraining Script**:
    ```powershell
    python retrain_sakhi.py
    ```
    **Note**: This script now automatically loads Sakhi's existing "brain" and trains her **further** on your new data. It does **not** erase what she already knows!

### 3. 📚 Large Scale Learning
To teach her a massive amount of information (like a whole book or a new language):
*   Paste the entire text into `indian_chat.txt`.
*   Make sure there are some "User:" and "Sakhi:" labels scattered throughout so she understands the conversational structure.
*   Run `python retrain_sakhi.py`.

### 💡 Pro Tips for Training:
*   **Repetition**: If she is struggling to remember a specific fact, repeat that pair 5-10 times in `indian_chat.txt`. It makes the signal stronger.
*   **Clean Data**: Make sure there are no typos in your training file. She learns exactly what she sees!
*   **Consistency**: Use the same tone in your manual entries ("Namaste", "ji", etc.) to keep her personality consistent.

**Happy Teaching! 🇮🇳🌸**
