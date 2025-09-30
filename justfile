
sim:
    uv run python -m jurigged -m simulator

fix:
    uvx ruff check --fix
    uvx ruff check --select I --fix .
    uvx ruff format