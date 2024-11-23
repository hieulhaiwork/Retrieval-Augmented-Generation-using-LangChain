import os

import tkinter as tk
from tkinter import filedialog

from chatbot import SimpleChatbot
from vector_db import VectorDBGenerator

def select_file(mode):
    root = tk.Tk()
    root.withdraw()
    if mode == 'file':
        data_path = filedialog.askopenfilename(title="Choose your file to retrieve.", 
                                                filetypes=[("All files", "*.*")])
    elif mode == 'folder':
        data_path = filedialog.askdirectory(title="Choose your file to retrieve.")

    return data_path

def main():
    print("=" * 50)
    print("Welcome to Simple Chatbot!".center(50))
    print("[1]. Add new database.")
    print("[2]. Load database.")
    print("[3]. Chat.")
    print("[4]. Exit.")
    print("=" * 50)

    while True:
        user_input = str(input("Human: "))

        if user_input.lower() in ['exit', 'bye', 'escape', '4']:
            print("Chatbot: Goodbye, see you again.")
            break
        
        elif user_input.lower() in ['new', 'add', '1']:
            data_path = select_file(mode='file')
            print("Creating your vector database...")
            db_generator = VectorDBGenerator(db_path="./chroma", data_path=data_path, embedding_model="models/embedding-001")
            file_name, file_extension = os.path.splitext(os.path.basename(data_path))
            vector_db = db_generator.generate_vectordb(db_name=file_name)
            print("Successfully create vector database.")

        elif user_input.lower() in ['load', '2']:
            data_path = select_file(mode='folder')
            print("Loading your vector database...")
            db_generator = VectorDBGenerator(db_path="./chroma", data_path=data_path, embedding_model="models/embedding-001")
            vector_db = db_generator.load_vectordb()
            print("Successfully load vector database.")

        elif user_input.lower() in ['ask', 'chat', '3']:
            assert vector_db is not None
            chatbot = SimpleChatbot(vector_db)
            for num_q in range(5):
                question = input("Human: ")
                response = chatbot.qachatbot(question)
                print(f"Chatbot: {response}")
        else:
            print("Invalid input!")

    print("End Chatbot...")
    print("=" * 50)


if __name__ == "__main__":
    main()
        
    
