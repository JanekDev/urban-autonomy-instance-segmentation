FROM python:3.9

RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

WORKDIR /workspace

COPY requirements.txt app.py demo.jpeg /workspace/

RUN pip install -r /workspace/requirements.txt

CMD ["python", "/workspace/app.py"]
