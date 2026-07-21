FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data
COPY templates ./templates
COPY static ./static
COPY app.py ./

RUN python -m student_performance.train

EXPOSE 5000

CMD ["python", "app.py"]
