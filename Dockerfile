FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
# API:        docker run -p 8000:8000 --env-file .env <image>
# Streamlit:  docker run -p 8501:8501 --env-file .env <image> streamlit run app.py --server.address 0.0.0.0
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
