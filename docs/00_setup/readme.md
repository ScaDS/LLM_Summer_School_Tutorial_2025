# Setting up your computer

This chapter provides instructions for setting up your computer.

## Setting up Python and Conda environments

When working with Python, we will make use of many plugins and software libraries which need to be organized.
One way of doing this, is by managing *Conda* environments.
A conda environment can be seen as a virtual desktop, or virtual computer, accessible via the terminal. 
If you install some software into one Conda environment, it may not be accessible from another environment. 
If a Conda environment breaks, e.g. incompatible software was installed, you can just make a new one and start over.

See also
* [Managing Scientific Python environments using Conda, Mamba and friends](https://focalplane.biologists.com/2022/12/08/managing-scientific-python-environments-using-conda-mamba-and-friends/)
* [Scientific Data Analysis with Python](https://youtu.be/MOEPe9TGBK0)
* [Getting started with Mambaforge and Python](https://biapol.github.io/blog/mara_lampert/getting_started_with_mambaforge_and_python/readme.html)

## Install Mini-Forge
Download and install miniforge. We recommend the distribution [miniforge of conda](https://conda-forge.org/download/). If you already have an old [Ana]conda installation you haven't touched for a while, it is recommended to uninstall it and install mini-forge instead.

For ease-of-use, it is recommended to install it for your use only and to add Conda to the PATH variable during installation.

![img.png](miniforge1.png)

![img.png](miniforge2.png)

## Setting up a conda environment

You can create a conda environment using this commands from the terminal.

```
conda env create -f https://raw.githubusercontent.com/ScaDS/LLM_Summer_School_Tutorial_2025/main/docs/00_setup/environment.yml
```

### Activating the environment
Activate the environment:
```
conda activate scads-llm
```

## Setting up VPN to TU-Dresden

You will need a VPN-connection to TU Dresden. Install everything necessary as explained on [this page](https://tu-dresden.de/zih/dienste/service-katalog/arbeitsumgebung/zugang_datennetz/vpn) ([eduVPN recommended](https://faq.tickets.tu-dresden.de/otrs/public.pl?Action=PublicFAQZoom;ItemID=1284). If you need a Guest Account at TU Dresden, please reach out to the ScaDS.AI Training School Coordination.


## Setting up API keys

For executing the exercises in this notebook collection, you will get a ScaDS.AI API Key: [download ZIP](api_key.zip); the password will be provided on-site. If you want to use the notebooks later on, you need to get your own ScaDS.AI API key. Some notebooks will also work with [Kisski/GWDG](https://kisski.gwdg.de/en/leistungen/2-02-llm-service/) and [Blablador/Helmholtz](https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/) API Keys. These services are free to use for Germen academics.

You can then save these keys in the environment variables, e.g. as `SCADSAI_API_KEY`, `BLABLADOR_API_KEY` and/or `KISSKI_API_KEY` of your computer as explained [on this page](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).


## Installing ollama

Optional: To make use of the ollama-based local models, please install [ollama](https://ollama.com/download). The notebooks in this folder were tested with ollama version 0.5.7. For this, it is recommended to use a computer with an NVidia Graphics Card.

Consider downloading these open-weight models to  run them locally:

* [mistral:v0.3](https://ollama.com/library/mistral:v0.3),
* [gemma3:1b](https://ollama.com/library/gemma3:1b)
* [gemma3:4b](https://ollama.com/library/gemma3:4b)


You can do this by running these commands:
```
ollama run mistral:v0.3
ollama run gemma3:1b
ollama run gemma3:4b

```

Note: You can print out which models you have downloaded like this:
```
ollama list
```
