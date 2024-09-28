import logging
from typing import List

from fastapi import FastAPI, UploadFile, File, Body
from fastapi import HTTPException
from starlette.responses import JSONResponse

from webapp.ml_model import predict_price as predict_price_model
from webapp.pipeline import import_csv_file
from webapp.schemas import PredictPrice

# Inicializando o aplicativo FastAPI
app = FastAPI(root_path="/tech-challenge-fase-3")
logging.basicConfig(level=logging.INFO)


@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """
    Endpoint para processar e ingerir os dados do arquivo CSV.

    Este endpoint aceita arquivos CSV para processamento. O arquivo CSV deve ser enviado como multipart/form-data.

    Exemplo de arquivo CSV disponível em: `/resources/diamonds/diamonds.csv`.

    Args:
        file (UploadFile): Arquivo CSV a ser processado.

    Returns:
        JSONResponse: Resposta JSON contendo o resultado da avaliação.
    """
    # Verificar se o tipo MIME é CSV
    if file.content_type != "text/csv":
        logging.error(f"Tipo de arquivo inválido: {file.content_type}")
        raise HTTPException(status_code=400, detail="O arquivo deve ser do tipo CSV (text/csv).")

    try:
        eval_result = import_csv_file(file)
        logging.info("Arquivo CSV enviado e processado com sucesso.")
        return JSONResponse(status_code=200, content=eval_result)
    except Exception as e:
        logging.error(f"Erro ao processar o arquivo: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/predict-price")
async def predict_price_endpoint(predict_price_input: List[List[PredictPrice]] = Body(
    ...,
    example=[
        [
            {
                "quilate": 0.23,
                "lg_topo": 55.0,
                "x": 3.95,
                "y": 3.98,
                "z": 2.43
            }
        ]
    ]
)) -> JSONResponse:
    """
    Endpoint para prever o preço com base nas características fornecidas.

    Args:
        predict_price_input (List[List[PredictPrice]]): Dados de entrada para previsão.

    Returns:
        JSONResponse: Resposta JSON contendo o preço previsto.
    """
    try:
        preco = predict_price_model(predict_price_input)
        logging.info("Previsão de preço realizada com sucesso.")
        return JSONResponse(status_code=200, content={"preco": preco.tolist()})
    except Exception as e:
        logging.error(f"Erro ao realizar a previsão de preço: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
