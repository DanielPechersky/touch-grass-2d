
sim:
    uv run python -m jurigged -m simulator

profile:
    uv run python -m cProfile -o profile.prof -m simulator

fix:
    uvx ruff check --fix
    uvx ruff check --select I --fix .
    uvx ruff format