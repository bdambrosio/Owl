# Owl ![alt text](https://github.com/bdambrosio/Owl/images/Owl.jpg?raw=true)
Owl is my experimental Research Assistant / general support / research project on Language-based cognitive architecture. 
* Owl has a UI, so you can chat with it. Complex context management include conversation history management, topic-based working memory, sentiment analysis, background reflection, even respond in speech if you like.
* Owl has a long term topic/subtopic memory, updated offline when it isn't busy.
* Owl can search the web (using a port of my llmsearch project), a local faiss of wikipedia, SemanticScholar, and it's own cache of research papers, all as part of chat or in support of other tasks.
* Owl's research RAG is multi-level - extensive analysis/rewrite occurs at paper ingest.
* Owl can write extended research reports: query to understand topic, generate outline, search SemanticScholar, retrieve and index papers, and write and refine a multi-page research report, with references. This process can take quite some time, depending on how the report task is configured
* Owl also has an active working memory it can do semantic retrieval from (work in progress)
    ** with full CRUD
* Owl can also create hierarchical plan structures, including references to working memory variables (plan construction and execution are works in progress)
* Owl can resolve the tasks in those plans to subplans rooted at a set of text operations performed by LLM or function call.

Owl is mostly useful at  the moment, if at all, as bits of example code. There is NO documentation on how to install, and it isn't built for easy installation. It uses multiple servers: 
* a fastapi-based llm server for local-hosted llms (although you can choose to use OpenAI or MistralAI if you have keys - I often configure report writing to use gpt-4 or mistral-medium for final report generation). I usually use mixtral-8x7B-instruct 6bit exl2 on a pair of 4090's, fast enough chatting is quite comfortable.
* a fastapi-based web search server (you need a google api key to run this one)
* a fastapi-based indexing server (so paper-indexing happens asynchonously, server handles queueing of multiple index requests)
* a grobid server for parsing pdfs (this one is easy, just a stock grobid container)
* a simple tts server (again trivial, but still, one more server to start, only if you want speech)

Owl is built on Alphawave-py / promptrix-py, both heavily modded. Modded Alphawave-py is included, but promptrix-py isn't, I'll fix that soon, but it should work with stock promptrix_py.

Disclaimer: my emphasis at the moment is on quality, not speed. Owl can spend time 'thinking' sometimes.

With that disclaimer, enough of Owl has moved from development mode to daily use mode that I will be expanding documentation in the near future. 

