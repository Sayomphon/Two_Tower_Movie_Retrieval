"""Entry point ของ Gradio demo — ใช้ได้ทั้ง local และ Hugging Face Spaces

    pip install -e ".[demo]"
    movie-retrieval all        # สร้าง artifacts ก่อน (ถ้ายังไม่มี)
    python app.py

logic ทั้งหมดอยู่ใน `movie_retrieval.demo` เพื่อให้ test ได้และไม่ผูกกับ hosting platform
"""

from movie_retrieval.demo import main

if __name__ == "__main__":
    main()
