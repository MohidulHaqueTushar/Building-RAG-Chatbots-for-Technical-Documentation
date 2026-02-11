# Building RAG Chatbots for Technical Documentation

This project demonstrates the implementation of a Retrieval-Augmented Generation (RAG) chatbot capable of answering questions about technical documentation. The proof-of-concept is a context-aware car assistant that explains dashboard warnings using a car manual.

## Project Overview

### Purpose

The primary goal of this project is to build an intelligent chatbot that can understand and answer user queries based on a specific set of documents. It showcases how to augment a large language model with external knowledge, enabling it to provide accurate answers from a given technical manual instead of relying solely on its pre-trained data.

### What It Achieves

This project successfully builds a complete, offline RAG pipeline that:
- Ingests and processes technical documentation from an HTML file.
- Generates vector embeddings from the document content and stores them for efficient retrieval.
- Responds to a user's natural language question with a concise and relevant answer sourced from the document.
- Converts the generated text answer into speech, demonstrating a use-case for a hands-free interface.

## Real-World Applications

This technology can be adapted for numerous real-world scenarios:

-   **In-Vehicle Assistants**: Provide drivers with hands-free access to information about their vehicle, from warning lights to feature explanations.
-   **Customer Support Automation**: Power chatbots that can answer customer questions by drawing from product manuals and knowledge bases, reducing wait times.
-   **Corporate Knowledge Management**: Enable employees to quickly search through internal documentation, reports, and wikis.
-   **Educational Tools**: Create interactive study aids that help students understand complex topics by answering questions based on textbook material.

## Technology Stack

This project utilizes open-source and local technologies to ensure privacy and cost-effectiveness.

-   **Core Framework**: **LangChain** is used to orchestrate the entire RAG pipeline, connecting the different components in a modular way.
-   **Language Model (LLM)**: **`google/flan-t5-base`**, a model from Hugging Face, runs locally for answer generation. This avoids reliance on external APIs and associated costs.
-   **Embedding Model**: **`sentence-transformers/all-MiniLM-L6-v2`** is used for creating text embeddings. Its efficiency makes it suitable for running on local CPU.
-   **Vector Store**: **Chroma DB** serves as the vector database to store and retrieve document embeddings quickly.
-   **Document Loader**: **UnstructuredHTMLLoader** is used to parse the content from the source HTML file.
-   **Text-to-Speech**: **pyttsx3** provides the functionality to convert the chatbot's text response into audible speech.