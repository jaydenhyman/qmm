
# Getting started

## Install the uv package manager

Windows

```sh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Mac and Linux

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Running the server

Install Python, dependencies and run the server:

```sh
cd "path/to/qmm-server/folder"
uv python install 3.10
uv python pin 3.10
uv sync
uv run server
```

Open *Digraph Builder* sidebar and refresh window.

## Using Jupyter

Install Python, dependencies, Jupyter, create a kernel and run Jupyter Lab:

```sh
cd "path/to/qmm-server/folder"
uv python install 3.10
uv python pin 3.10
uv sync
uv pip install qmm-core
uv run python -m ipykernel install --user --name qmm --display-name "QMM Environment"
uv run jupyter lab
```

When creating a new notebook, select the "QMM Environment" kernel.
