FROM python:3.12-alpine3.19

# Instalar as dependências de compilação
RUN apk add --no-cache gcc g++ musl-dev linux-headers

# Definir o diretório de trabalho
WORKDIR /app

# Copiar todos os arquivos do diretório atual para o diretório de trabalho no contêiner
COPY . .

# Instalar as dependências do Python
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Definir o entrypoint do contêiner
ENTRYPOINT ["python3", "-m", "uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
