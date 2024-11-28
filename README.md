# RAG - Retrieval-Augmented Generation

## How to install

1. Clone this repo

```
git clone git@github.com:pepcmarques/simple-rag.git 
```

2. Create a virtual environment (optional)

```
python -m venv .venv
```

3. Activate the virtual environment (optional - requires item 2)

```
source .venv/bin/activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Download the `synthia-7b-v2.0-16k.Q4_K_M.gguf` model

```
pip install huggingface-hub  # if not installed yet
huggingface-cli download TheBloke/SynthIA-7B-v2.0-16k-GGUF synthia-7b-v2.0-16k.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

Further details [here](https://huggingface.co/TheBloke/SynthIA-7B-v2.0-16k-GGUF)

## How to use it

1. Copy your PDF files to the `data`, or keep the examples there

2. Populate the RAG database with the documents

```
python populate_database.py [--reset]
```

> **reset** - optional parameter for deleting the RAG database

3. Run a query

```
python query_data.py "How to get out of the jail in monopoly?"
```

**Note**: Thanks for Pixegami and contributors (https://github.com/pixegami/rag-tutorial-v2)
