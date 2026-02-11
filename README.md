# Building-RAG-Chatbots-for-Technical-Documentation

Building RAG Chatbots for Technical Documentation
Implement retrieval augmented generation (RAG) with LangChain to create a chatbot for answering questions about technical documentation.


Project Description
You'll create a context-aware chatbot by integrating a car manual with an LLM using LangChain and Retrieval Augmented Generation (RAG). The goal is to create a car assistant that can explain dashboard warnings and recommend actions while driving. Say goodbye to boring manuals!
Project Instructions
The car manual HTML document has been loaded for you as car_docs. Using Retrieval Augmented Generation (RAG) to make OpenAI's gpt-4o-mini aware of the contents of car_docs, answer the following user query:

"The Gasoline Particular Filter Full warning has appeared. What does this mean and what should I do about it?"

Store the answer to the user query in the variable answer.
Submissions and help
How to approach the project
Steps to complete

1
Split the document
2
Store the embeddings
3
Create a retriever

![A car dashboard with lots of new technical features.](images/dashboard.jpg)

You're working for a well-known car manufacturer who is looking at implementing LLMs into vehicles to provide guidance to drivers. You've been asked to experiment with integrating car manuals with an LLM to create a context-aware chatbot. They hope that this context-aware LLM can be hooked up to a text-to-speech software to read the model's response aloud.

As a proof of concept, you'll integrate several pages from a car manual that contains car warning messages and their meanings and recommended actions. This particular manual, stored as an HTML file, `mg-zs-warning-messages.html`, is from an MG ZS automobile, a compact SUV. Armed with your newfound knowledge of LLMs and LangChain, you'll implement Retrieval Augmented Generation (RAG) to create the context-aware chatbot.

**Note: Although we'll be using the OpenAI API in this project, you do not need to specify an API key.**