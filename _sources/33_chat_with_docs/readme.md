# Retreivela Augmented Generation

This folder contains notebooks demonstrating different strategies for interacting with documents using a chatbot and [Retrieval Augmented Generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation). In one example, we will use the [HPC compendium](https://compendium.hpc.tu-dresden.de/) which is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by TU Dresden ZIH. 

## Chat with PDFs
There is also simple Chat interface based on [voila](https://github.com/voila-dashboards/voila) and [llama-index](https://github.com/run-llama/llama_index) that allows you to query information from documents using human language.

You can start the chat using this terminal command:

```
voila chat-with-docs.ipynb
```

![](docs/screenshot.png)

Some of the code in this folder was inspired by the tutorial [here](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/).
