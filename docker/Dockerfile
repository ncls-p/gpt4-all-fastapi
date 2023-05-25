FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "src/app.py"]