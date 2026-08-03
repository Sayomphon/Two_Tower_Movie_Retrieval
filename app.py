"""Entry point for the Gradio demo — works both locally and on Hugging Face Spaces

    pip install -e ".[demo]"
    movie-retrieval all        # build the artifacts first (if they do not exist yet)
    python app.py

All the logic lives in `movie_retrieval.demo` so it stays testable and unbound to any
hosting platform
"""

from movie_retrieval.demo import main

if __name__ == "__main__":
    main()
