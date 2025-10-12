# GenAi-Service

## Installation

Set up the project environment with Conda and install dependencies in this order:

```bash
conda create -n genaiservice python=3.11
conda activate genaiservice
pip install -r requirements.txt
```

## Running the services

Start the FastAPI backend with the built-in development server:

```bash
fastapi dev main.py
```

In a separate terminal, launch the Streamlit client:

```bash
streamlit run client.py
```

By default the FastAPI app listens on `http://127.0.0.1:8000` and Streamlit opens in your browser at `http://localhost:8501`.

