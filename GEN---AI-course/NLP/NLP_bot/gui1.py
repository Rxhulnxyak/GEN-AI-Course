import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import webbrowser
import datetime
import threading

try:
    import wikipedia
except ImportError:
    wikipedia = None


class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        self.setup_gui()
        self.load_chatbot_data()
        self.conversation_history = []
        self.is_dark_mode = False

    def setup_gui(self):
        self.master.title("Advanced NLP Chatbot")
        self.master.geometry("550x650")
        self.master.configure(bg="#f0f0f0")

        style = ttk.Style()
        style.theme_use("clam")

        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_history = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, width=60, height=25, font=("Arial", 10)
        )
        self.chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_history.config(state=tk.DISABLED)

        # Status Label for "Typing..." indicator
        self.status_var = tk.StringVar()
        self.status_var.set("Bot is Online")
        self.status_label = tk.Label(main_frame, textvariable=self.status_var, font=("Arial", 9, "italic"), bg="#f0f0f0", fg="gray")
        self.status_label.pack(anchor=tk.W, pady=(0, 5))

        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)

        self.user_input = ttk.Entry(input_frame, width=50, font=("Arial", 10))
        self.user_input.pack(side=tk.LEFT, padx=(0, 5), expand=True, fill=tk.X)
        self.user_input.bind("<Return>", lambda event: self.send_message())

        self.send_button = ttk.Button(
            input_frame, text="Send", command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.clear_button = ttk.Button(
            button_frame, text="Clear Chat", command=self.clear_chat
        )
        self.clear_button.pack(side=tk.LEFT, padx=(0, 5))

        self.save_button = ttk.Button(
            button_frame, text="Save Chat", command=self.save_chat
        )
        self.save_button.pack(side=tk.LEFT, padx=(0, 5))

        self.theme_button = ttk.Button(
            button_frame, text="Toggle Theme", command=self.toggle_theme
        )
        self.theme_button.pack(side=tk.LEFT)

        self.help_button = ttk.Button(button_frame, text="Help", command=self.show_help)
        self.help_button.pack(side=tk.RIGHT)

    def load_chatbot_data(self):
        self.lemmatizer = WordNetLemmatizer()
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        intents_path = os.path.join(base_dir, "Data", "intents.json")
        with open(intents_path, encoding='utf-8') as file:
            self.intents = json.loads(file.read())
        self.words = pickle.load(open("words.pkl", "rb"))
        self.classes = pickle.load(open("classes.pkl", "rb"))
        self.model = load_model("chatbot_model.h5")

    def send_message(self):
        user_message = self.user_input.get().strip()
        self.user_input.delete(0, tk.END)
        if user_message:
            self.update_chat_history(f"You: {user_message}", "user")
            
            # Show typing indicator
            self.status_var.set("Bot is typing...")
            self.master.update_idletasks()
            
            # Use after() to simulate delay and not freeze GUI
            self.master.after(600, lambda: self.process_bot_response(user_message))

    def process_bot_response(self, user_message):
        bot_response = self.get_bot_response(user_message)
        self.update_chat_history(f"Bot: {bot_response}", "bot")
        self.conversation_history.append((user_message, bot_response))
        self.status_var.set("Bot is Online")

    def update_chat_history(self, message, sender="bot"):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, message + "\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        if self.is_dark_mode:
            bg_color = "#2b2b2b"
            fg_color = "#ffffff"
            input_bg = "#3c3f41"
            status_bg = "#2b2b2b"
        else:
            bg_color = "#f0f0f0"
            fg_color = "#000000"
            input_bg = "#ffffff"
            status_bg = "#f0f0f0"
            
        self.master.configure(bg=bg_color)
        self.chat_history.config(bg=input_bg, fg=fg_color, insertbackground=fg_color)
        self.status_label.config(bg=status_bg)

    def get_bot_response(self, user_message):
        msg_lower = user_message.lower()
        if msg_lower in ["exit", "quit", "bye"]:
            return "Goodbye! Have a great day!"
        elif msg_lower.startswith("search "):
            query = user_message[7:]
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"I've opened a web search for '{query}'."
        elif msg_lower.startswith("wiki "):
            if wikipedia:
                query = user_message[5:]
                try:
                    # Fetching 2 sentences summary from Wikipedia
                    summary = wikipedia.summary(query, sentences=2)
                    return f"Here is what I found on Wikipedia:\n{summary}"
                except Exception as e:
                    return f"Sorry, I couldn't find a clear Wikipedia page for '{query}'."
            else:
                return "The 'wikipedia' library is not installed. Please install it using 'pip install wikipedia'."
        elif msg_lower == "time":
            return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."
        else:
            ints = self.predict_class(user_message)
            return self.get_response(ints)

    def clean_up_sentence(self, sentence):
        return [
            self.lemmatizer.lemmatize(word.lower())
            for word in nltk.word_tokenize(sentence)
        ]

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [1 if word in sentence_words else 0 for word in self.words]
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [
            {"intent": self.classes[r[0]], "probability": str(r[1])} for r in results
        ]

    def get_response(self, intents_list):
        if not intents_list:
            return "I'm not sure how to respond to that. Can you please rephrase your question?"
        tag = intents_list[0]["intent"]
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "I'm sorry, I don't have a specific response for that. Can you try asking something else?"

    def clear_chat(self):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.config(state=tk.DISABLED)
        self.conversation_history.clear()

    def save_chat(self):
        filename = "chat_history.txt"
        with open(filename, "a") as f: # changed to append so it doesn't delete old chats
            for user_msg, bot_msg in self.conversation_history:
                f.write(f"You: {user_msg}\n")
                f.write(f"Bot: {bot_msg}\n\n")
        messagebox.showinfo("Chat Saved", f"Chat history has been saved to {filename}")

    def show_help(self):
        help_text = """
        Welcome to the Advanced NLP Chatbot!

        Special Commands:
        - Type 'exit', 'quit', or 'bye' to end the conversation.
        - Type 'search <query>' to open a Google web search.
        - Type 'wiki <topic>' to fetch a summary directly from Wikipedia.
        - Type 'time' to get the current time.

        Features:
        - Toggle Theme: Switch between Light and Dark mode.
        - Clear Chat: Clears the current conversation.
        - Save Chat: Saves the conversation history to a file.

        Enjoy chatting!
        """
        messagebox.showinfo("Chatbot Help", help_text)


if __name__ == "__main__":
    root = tk.Tk()
    ChatbotGUI(root)
    root.mainloop()
