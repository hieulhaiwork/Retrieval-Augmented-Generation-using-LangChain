<div align='center'>
  <h1> Retrieval-Augmented Generation using LangChain </h1>
  <a>
    <img src="https://blogs.nvidia.com/wp-content/uploads/2023/11/NVIDIA-RAG-diagram-scaled.jpg">
    <p> Source: https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/ </p>
  </a>
  <a href=""> Demo </a>
</div>
</br>
</br>
<h3> Description </h3>
<p> This is my personal project, applying RAG to enhance the performance of LLM model in QA tasks.<br>
    The user can upload .pdf file then chat with the chatbot to discover information in the file: ask for information, summarize text, generate questions,....</p>

<h3> Run </h3>
<h5> 1. Installing packages </h5>
 The requirements.txt file will list all Python libraries that are necessary for this project. Ensure that those libraries has been fully installed in your environment.<br>
 Or you can run this script:  
 
```
pip install -r requirements.txt
```

<h5> 2. Create .env file </h5>
 Create .env file to store env variables including:
 
```
 GOOGLE_API_KEY = "your keys"
```

<h5> 3. Run </h5>
 First, run `vector_database.py` file 


