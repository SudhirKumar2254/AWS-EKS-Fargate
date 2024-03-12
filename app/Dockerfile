FROM python:3.8-slim as builder
WORKDIR /usr/src/app
COPY . .
RUN pip install --user flask

FROM python:3.8-slim as app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/src/app .

ENV PATH=/root/.local:$PATH
EXPOSE 5000

CMD ["python3", "main.py"]