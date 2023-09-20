FROM python:3.8-slim-bullseye
WORKDIR /app
COPY . /app

RUN apt update -y 

RUN pip install -r requirements.txt
CMD ["python3", "app.py"]