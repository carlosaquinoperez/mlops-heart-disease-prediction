FROM python:3.12.1-slim-bookworm

# This copies the 'uv' binary without installing all its build tools.
COPY --from=astral/uv:latest /uv /usr/bin/uv

# 3. Set the working directory inside the container
WORKDIR /app

# This lets us run 'uvicorn' directly without 'uv run'.
ENV PATH="/app/.venv/bin:$PATH"

# (Docker caches this layer. It won't re-install dependencies unless these files change.)
COPY pyproject.toml uv.lock ./

# This installs all packages from 'uv.lock' into the .venv
RUN uv sync --locked

# (Our model, our API script, and our helper script)
COPY model.bin predict.py helpers.py ./

# Expose the port that FastAPI will run on [cite: 395]
EXPOSE 9696

# Define the command to run when the container starts
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]