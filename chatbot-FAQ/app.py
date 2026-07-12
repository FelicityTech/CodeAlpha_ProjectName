"""
Simple chat UI for the FAQ chatbot, built with Gradio.

Run: python app.py
Then open the local URL it prints (usually http://127.0.0.1:7860).
"""
import gradio as gr

from chatbot import FAQChatbot

bot = FAQChatbot()

SUGGESTED_QUESTIONS = [
    "How do I reset my password?",
    "How much does CloudSync cost?",
    "Is my data encrypted?",
    "How do I cancel my subscription?",
]


def respond(message, history):
    return bot.respond(message)


with gr.Blocks(title="CloudSync FAQ Assistant") as demo:
    gr.Markdown(
        """
        # 💬 CloudSync FAQ Assistant
        Ask a question about billing, your account, syncing, or security.
        I'll match it against our FAQ database using TF-IDF + cosine similarity.
        """
    )

    chatbot_ui = gr.ChatInterface(
        fn=respond,
        examples=SUGGESTED_QUESTIONS,
        chatbot=gr.Chatbot(height=420, placeholder="Ask me something about CloudSync..."),
        textbox=gr.Textbox(placeholder="Type your question here...", scale=7),
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="teal"))
