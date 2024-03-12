FROM python:3.11
WORKDIR /app

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY . /app

ENV PATH=/app:$PATH

EXPOSE 5000

CMD ["python3", "main.py"]