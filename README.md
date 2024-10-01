# leniolabs_challenge
This is the challenge to get into a Data Scientist position in Leniolabs. 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## The challenge was this:

**Case Study**

The goal is to develop a simple chatbot with RAG using a dataset of your choice (FAQs, articles, papers, product descriptions, literature…) and create an architecture diagram that includes both the implemented components, as well as more advanced features that could be added, such as a memory mechanism, guardrails, re-ranker (although these features do not need to be implemented).
The idea is to make a simple demo in an upcoming meeting, which can be done through notebooks, it is not necessary to implement a UI.

**Specific tasks:**

Develop a chatbot with RAG: A simple chatbot that accepts queries, connects to a vector database to bring relevant information from a dataset and generates responses using some LLM. Implement a retrieval mechanism based on embeddings using a vector database. Both the data, the LLM, the vector database, the embedding model, etc. are of your choice.
Create an architecture diagram: Include both the components you implement (data ingestion, embedding generation, vector database, retrieval process, and LLM integration) and the advanced components (memory, guardrails, and re-ranker).

**Presentation:**

Demo the chatbot in an upcoming call. Explanation of both implemented and unimplemented concepts in the architecture diagram. Explain how you would integrate more advanced features into the chatbot. You don't need to send any code or written documentation; the live demo and discussion will suffice.

**Deadline:**
One week

**The goal** 
Develop a basic, working RAG and discuss how more advanced features could improve it, not so much the quality of the code.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Roadmap

**What was done in the first POC and the deliver for the challenge was this:**

1. Training a chatbot that uses a basic RAG architecture to extract context from books and complement the answers it delivers by using LLM's (gpt-4o-mini in this case)
2. Automatic evaluation mechanism with an evaluator agent and using a guideline created by me
2. A version with a functional fast API
3. Credential management with YAML

**However, i will use this project as starting point in my learning on how to deploy a System that use LLM's into production and i will continue to work on it during 2024.** 

**My next step are:**

1. Decouple training, evaluation and prediction phases
2. Add a parser that can read complex documents and tables (probably llama-parser)
3. Improve the evaluation mechanism
4. Incorporate a tooling tool

## The Process diagram:

![Chatbot Architecture with LLM and RAG System (1)](https://github.com/user-attachments/assets/348a9a89-0dac-43d8-aa55-6a4430df0f17)
