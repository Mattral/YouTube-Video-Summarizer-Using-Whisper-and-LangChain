# YouTube-Video-Summarizer-Using-Whisper-and-LangChain

 It is possible to effectively extract key takeaways from videos by leveraging Whisper to transcribe YouTube audio files and utilizing LangChain's summarization techniques, including stuff, refine, and map_reduce. We also highlighted the customizability of LangChain, allowing personalized prompts, multilingual summaries, and storage of URLs in a Deep Lake vector store. By implementing these advanced tools, you can save time, enhance knowledge retention, and improve your understanding of various topics. Enjoy the tailored experience of data storage and summarization with LangChain and Whisper.

The following diagram explains what we are going to do in this project.

<img align="center" src="ytvs.avif" alt="banner">

First, we download the youtube video we are interested in and transcribe it using Whisper. Then, we’ll proceed by creating summaries using two different approaches:

- First we use an existing summarization chain to generate the final summary, which automatically manages embeddings and prompts.
- Then, we use another approach more step-by-step to generate a final summary formatted in bullet points, consisting in splitting the transcription into chunks, computing their embeddings, and preparing ad-hoc prompts.


## Introduction

In the digital era, the abundance of information can be overwhelming, and we often find ourselves scrambling to consume as much content as possible within our limited time. YouTube is a treasure trove of knowledge and entertainment, but it can be challenging to sift through long videos to extract the key takeaways. Worry not, as we've got your back! In this repo, we will unveil a powerful solution to help you efficiently summarize YouTube videos using two cutting-edge tools: Whisper and LangChain.

Workflow:
- Download the YouTube audio file.
- Transcribe the audio using Whisper.
- Summarize the transcribed text using LangChain with three different approaches: stuff, refine, and map_reduce.
- Adding multiple URLs to DeepLake database, and retrieving information. 
Installations:

Remember to install the required packages with the following command: pip install langchain deeplake openai tiktoken. Additionally, install also the yt_dlp and openai-whisper packages, which have been tested in this repo with versions  2023.6.21 and 20230314, respectively.

```
!pip install -q yt_dlp
!pip install -q git+https://github.com/openai/whisper.git
```

Then, we must install the ffmpeg application, which is one of the requirements for the yt_dlp package. This application is installed on Google Colab instances by default. The following commands show the installation process on Mac and Ubuntu operating systems.

```
# MacOS (requires https://brew.sh/)
#brew install ffmpeg

# Ubuntu
#sudo apt install ffmpeg
```

You can read the following article if you're working on an operating system that hasn't been mentioned earlier (like Windows). It contains comprehensive, step-by-step instructions on ["How to install ffmpeg.”](https://www.hostinger.com/tutorials/how-to-install-ffmpeg)

Next step is to add the API key for OpenAI and Deep Lake services in the environment variables. You can either use the load_dotenv function to read the values from a .env file, or by running the following code. Remember that the API keys must remain private since anyone with this information can access these services on your behalf.

```
import os

os.environ['OPENAI_API_KEY'] = "<OPENAI_API_KEY>"
os.environ['ACTIVELOOP_TOKEN'] = "<ACTIVELOOP_TOKEN>"
```

For this experiment, we have selected a video featuring Yann LeCun, a distinguished computer scientist and AI researcher. In this engaging discussion, LeCun delves into the challenges posed by large language models.

The download_mp4_from_youtube() function will download the best quality mp4 video file from any YouTube link and save it to the specified path and filename. We just need to copy/paste the selected video’s URL and pass it to mentioned function.

```
import yt_dlp

def download_mp4_from_youtube(url):
    # Set the options for the download
    filename = 'lecuninterview.mp4'
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': filename,
        'quiet': True,
    }

    # Download the video file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)

url = "https://www.youtube.com/watch?v=mBjPyte2ZZo"
download_mp4_from_youtube(url)
```

## Now it’s time for Whisper!

Whisper is a cutting-edge, automatic speech recognition system developed by OpenAI. Boasting state-of-the-art capabilities, Whisper has been trained on an impressive 680,000 hours of multilingual and multitasking supervised data sourced from the web.  This vast and varied dataset enhances the system's robustness, enabling it to handle accents, background noise, and technical language easily. OpenAI has released the models and codes to provide a solid foundation for creating valuable applications harnessing the power of speech recognition.

The whisper package that we installed earlier provides the .load_model() method to download the model and transcribe a video file. Multiple different models are available: tiny, base, small, medium, and large. Each one of them has tradeoffs between accuracy and speed. We will use the 'base' model for this tutorial.

```
import whisper

model = whisper.load_model("base")
result = model.transcribe("lecuninterview.mp4")
print(result['text'])
```

```
/home/cloudsuperadmin/.local/lib/python3.9/site-packages/whisper/transcribe.py:114: UserWarning: FP16 is not supported on CPU; using FP32 instead
warnings.warn("FP16 is not supported on CPU; using FP32 instead")

Hi, I'm Craig Smith, and this is I on A On. This week I talked to Jan LeCoon, one of the seminal figures in deep learning development and a long-time proponent of self-supervised learning. Jan spoke about what's missing in large language models and his new joint embedding predictive architecture which may be a step toward filling that gap. He also talked about his theory of consciousness and the potential for AI systems to someday exhibit the features of consciousness. It's a fascinating conversation that I hope you'll enjoy. Okay, so Jan, it's great to see you again. I wanted to talk to you about where you've gone with so supervised learning since last week's spoke. In particular, I'm interested in how it relates to large language models because they have really come on stream since we spoke. In fact, in your talk about JEPA, which is joint embedding predictive architecture. […and so on]

```
We’ve got the result in the form of a raw text and it is possible to save it to a text file.


```

with open ('text.txt', 'w') as file:  
    file.write(result['text'])
```

## Summarization with LangChain
We first import the necessary classes and utilities from the LangChain library.

```
from langchain import OpenAI, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
```

This imports essential components from the LangChain library for efficient text summarization and initializes an instance of OpenAI's large language model with a temperature setting of 0. The key elements include classes for handling large texts, optimization, prompt construction, and summarization techniques.

This code creates an instance of the RecursiveCharacterTextSplitter
 class, which is responsible for splitting input text into smaller chunks. 

```
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
```

It is configured with a chunk_size of 1000 characters, no chunk_overlap, and uses spaces, commas, and newline characters as separators. This ensures that the input text is broken down into manageable pieces, allowing for efficient processing by the language model.

We’ll open the text file we’ve saved previously and split the transcripts using .split_text() method.

```
from langchain.docstore.document import Document

with open('text.txt') as f:
    text = f.read()

texts = text_splitter.split_text(text)
docs = [Document(page_content=t) for t in texts[:4]]
```

Each Document object is initialized with the content of a chunk from the texts list. The [:4] slice notation indicates that only the first four chunks will be used to create the Document objects. 

```
from langchain.chains.summarize import load_summarize_chain
import textwrap

chain = load_summarize_chain(llm, chain_type="map_reduce")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)
```

```
Craig Smith interviews Jan LeCoon, a deep learning developer and proponent of self-supervised learning, about his new joint embedding predictive architecture and his theory of consciousness. Jan's research focuses on self-supervised learning and its use for pre-training transformer architectures, which are used to predict missing words in a piece of text. Additionally, large language models are used to predict the next word in a sentence, but it is difficult to represent uncertain predictions when applying this to video.

```

With the following line of code, we can see the prompt template that is used with the map_reduce technique. Now we’re changing the prompt and using another summarization method:

```
print( chain.llm_chain.prompt.template )
```

```
Write a concise summary of the following:\n\n\n"{text}"\n\n\n CONCISE SUMMARY:
```

The "stuff" approach is the simplest and most naive one, in which all the text from the transcribed video is used in a single prompt. This method may raise exceptions if all text is longer than the available context size of the LLM and may not be the most efficient way to handle large amounts of text. 

We’re going to experiment with the prompt below. This prompt will output the summary as bullet points.

```

prompt_template = """Write a concise bullet point summary of the following:


{text}


CONSCISE SUMMARY IN BULLET POINTS:"""

BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template, 
                        input_variables=["text"])

```


Also, we initialized the summarization chain using the stuff as chain_type and the prompt above.

```
chain = load_summarize_chain(llm, 
                             chain_type="stuff", 
                             prompt=BULLET_POINT_PROMPT)

output_summary = chain.run(docs)

wrapped_text = textwrap.fill(output_summary, 
                             width=1000,
                             break_long_words=False,
                             replace_whitespace=False)
print(wrapped_text)
```

```
- Jan LeCoon is a seminal figure in deep learning development and a long time proponent of self-supervised learning
- Discussed his new joint embedding predictive architecture which may be a step toward filling the gap in large language models
- Theory of consciousness and potential for AI systems to exhibit features of consciousness
- Self-supervised learning revolutionized natural language processing
- Large language models lack a world model and are generative models, making it difficult to represent uncertain predictions
```

Great job! By utilizing the provided prompt and implementing the appropriate summarization techniques, we've successfully obtained concise bullet-point summaries of the conversation.

In LangChain we have the flexibility to create custom prompts tailored to specific needs. For instance, if you want the summarization output in French, you can easily construct a prompt that guides the language model to generate a summary in the desired language.

The 'refine' summarization chain is a method for generating more accurate and context-aware summaries. This chain type is designed to iteratively refine the summary by providing additional context when needed. That means: it generates the summary of the first chunk. Then, for each successive chunk, the work-in-progress summary is integrated with new info from the new chunk.

```
chain = load_summarize_chain(llm, chain_type="refine")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)
```

```
Craig Smith interviews Jan LeCoon, a deep learning developer and proponent of self-supervised learning, about his new joint embedding predictive architecture and his theory of consciousness. Jan discusses the gap in large language models and the potential for AI systems to exhibit features of consciousness. He explains how self-supervised learning has revolutionized natural language processing through the use of transformer architectures for pre-training, such as taking a piece of text, removing some of the words, and replacing them with black markers to train a large neural net to predict the words that are missing. This technique has been used in practical applications such as contact moderation systems on Facebook, Google, YouTube, and more. Jan also explains how this technique can be used to represent uncertain predictions in generative models, such as predicting the missing words in a text, or predicting the missing frames in a video.
```

The 'refine' summarization chain in LangChain provides a flexible and iterative approach to generating summaries, allowing you to customize prompts and provide additional context for refining the output. This method can result in more accurate and context-aware summaries compared to other chain types like 'stuff' and 'map_reduce'.

## Adding Transcripts to Deep Lake

This method can be extremely useful when you have more data. Let’s see how we can improve our expariment by adding multiple URLs, store them in Deep Lake database and retrieve information using QA chain.

First, we need to modify the script for video downloading slightly, so it can work with a list of URLs.

```
import yt_dlp

def download_mp4_from_youtube(urls, job_id):
    # This will hold the titles and authors of each downloaded video
    video_info = []

    for i, url in enumerate(urls):
        # Set the options for the download
        file_temp = f'./{job_id}_{i}.mp4'
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': file_temp,
            'quiet': True,
        }

        # Download the video file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            title = result.get('title', "")
            author = result.get('uploader', "")

        # Add the title and author to our list
        video_info.append((file_temp, title, author))

    return video_info

urls=["https://www.youtube.com/watch?v=mBjPyte2ZZo&t=78s",
    "https://www.youtube.com/watch?v=cjs7QKJNVYM",]
vides_details = download_mp4_from_youtube(urls, 1)
```

And transcribe the videos using Whisper as we previously saw and save the results in a text file.

```
import whisper

# load the model
model = whisper.load_model("base")

# iterate through each video and transcribe
results = []
for video in vides_details:
    result = model.transcribe(video[0])
    results.append( result['text'] )
    print(f"Transcription for {video[0]}:\n{result['text']}\n")

with open ('text.txt', 'w') as file:  
    file.write(results['text'])
```

```
long text output..................
```

Then, load the texts from the file and use the text splitter to split the text to chunks with zero overlap before we store them in Deep Lake.

```
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the texts
with open('text.txt') as f:
    text = f.read()
texts = text_splitter.split_text(text)

# Split the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
texts = text_splitter.split_text(text)
```

Similarly, as before we’ll pack all the chunks into a Documents:

```

from langchain.docstore.document import Document

docs = [Document(page_content=t) for t in texts[:4]]

```

Now, we’re ready to import Deep Lake and build a database with embedded documents:

```
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "<YOUR-ACTIVELOOP-ORG-ID>"
my_activeloop_dataset_name = "langchain_course_youtube_summarizer"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db.add_documents(docs)
```


In order to retrieve the information from the database, we’d have to construct a retriever object.

```

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4
```

The distance metric determines how the Retriever measures "distance" or similarity between different data points in the database. By setting distance_metric to 'cos', the Retriever will use cosine similarity as its distance metric. Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. It's often used in information retrieval to measure the similarity between documents or pieces of text. Also, by setting 'k' to 4, the Retriever will return the 4 most similar or closest results according to the distance metric when a search is performed.

We can construct and use a custom prompt template with the QA chain. The RetrievalQA chain is useful to query similiar contents from databse and use the returned records as context to answer questions. The custom prompt ability gives us the flexibility to define custom tasks like retrieving the documents and summaizing the results in a bullet-point style.

```
from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of transcripts from a video to answer the question in bullet points and summarized. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Summarized answer in bullter points:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
```

Lastly, we can use the chain_type_kwargs argument to define the custom prompt and for chain type the ‘stuff’  variation was picked. You can perform and test other types as well, as seen previously.

```
from langchain.chains import RetrievalQA

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 chain_type_kwargs=chain_type_kwargs)

print( qa.run("Summarize the mentions of google according to their AI program") )
```

```
• Google has developed an AI program to help people with their everyday tasks.
• The AI program can be used to search for information, make recommendations, and provide personalized experiences.
• Google is using AI to improve its products and services, such as Google Maps and Google Assistant.
• Google is also using AI to help with medical research and to develop new technologies.
```

Of course, you can always tweak the prompt to get the desired result, experiment more with modified prompts using different types of chains and find the most suitable combination. Ultimately, the choice of strategy depends on the specific needs and constraints of your project. 

## Conclusion
When working with large documents and language models, it is essential to choose the right approach to effectively utilize the information available. We have discussed three main strategies: "stuff," "map-reduce," and "refine."

The "stuff" approach is the simplest and most naive one, in which all the text from the documents is used in a single prompt. This method may raise exceptions if all text is longer than the available context size of the LLM and may not be the most efficient way to handle large amounts of text.

On the other hand, the "map-reduce" and "refine" approaches offer more sophisticated ways to process and extract useful information from longer documents. While the "map-reduce" method can be parallelized, resulting in faster processing times, the "refine" approach is empirically known to produce better results. However, it is sequential in nature, making it slower compared to the "map-reduce" method.

By considering the trade-offs between speed and quality, you can select the most suitable approach to leverage the power of LLMs for your tasks effectively.
