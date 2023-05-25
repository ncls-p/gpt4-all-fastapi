FROM 3.11.3-alpine3.18

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "src/app.py"]