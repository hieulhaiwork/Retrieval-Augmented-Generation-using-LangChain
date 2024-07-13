import gradio as gr
import time
from vector_database import *


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot], label = "Clear")
    file = gr.UploadButton(label = "Upload a file")

    def response(message, history):
        pass


demo.launch()
