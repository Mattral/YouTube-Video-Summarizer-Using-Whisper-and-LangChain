# YouTube-Video-Summarizer-Using-Whisper-and-LangChain
YouTube Video Summarizer Using Whisper and LangChain

 It is possible to effectively extract key takeaways from videos by leveraging Whisper to transcribe YouTube audio files and utilizing LangChain's summarization techniques, including stuff, refine, and map_reduce. We also highlighted the customizability of LangChain, allowing personalized prompts, multilingual summaries, and storage of URLs in a Deep Lake vector store. By implementing these advanced tools, you can save time, enhance knowledge retention, and improve your understanding of various topics. Enjoy the tailored experience of data storage and summarization with LangChain and Whisper.

The following diagram explains what we are going to do in this project.

<img align="center" src="ytvs.avif" alt="banner">

First, we download the youtube video we are interested in and transcribe it using Whisper. Then, we’ll proceed by creating summaries using two different approaches:

- First we use an existing summarization chain to generate the final summary, which automatically manages embeddings and prompts.
- Then, we use another approach more step-by-step to generate a final summary formatted in bullet points, consisting in splitting the transcription into chunks, computing their embeddings, and preparing ad-hoc prompts.


## Introduction

In the digital era, the abundance of information can be overwhelming, and we often find ourselves scrambling to consume as much content as possible within our limited time. YouTube is a treasure trove of knowledge and entertainment, but it can be challenging to sift through long videos to extract the key takeaways. Worry not, as we've got your back! In this lesson, we will unveil a powerful solution to help you efficiently summarize YouTube videos using two cutting-edge tools: Whisper and LangChain.

Workflow:
- Download the YouTube audio file.
- Transcribe the audio using Whisper.
- Summarize the transcribed text using LangChain with three different approaches: stuff, refine, and map_reduce.
- Adding multiple URLs to DeepLake database, and retrieving information. 
Installations:

Remember to install the required packages with the following command: pip install langchain deeplake openai tiktoken. Additionally, install also the yt_dlp and openai-whisper packages, which have been tested in this lesson with versions  2023.6.21 and 20230314, respectively.
