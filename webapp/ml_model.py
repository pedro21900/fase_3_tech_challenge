from typing import List
import joblib
import numpy as np
from webapp.schemas import PredictPrice


def predict_price(independent_vars: List[List[PredictPrice]]) -> np.ndarray:
    """
    Função para prever o preço usando um modelo previamente treinado e salvo,
    aplicando escalonamento de dados e invertendo a padronização para o valor real.

    Args:
        independent_vars (List[List[PredictPrice]]): Lista de objetos PredictPrice
         com os atributos necessários para a previsão.

    Returns:
        np.ndarray: Previsão do preço em valores reais.
    """
    # Carregar o modelo salvo
    model_path = './resources/ml/modelo_rf.pkl'
    scaler_x_path = './resources/ml/scaler_X.pkl'
    scaler_y_path = './resources/ml/scaler_y.pkl'

    loaded_model = joblib.load(model_path)
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    # Extrair as características dos objetos PredictPrice
    independent_vars = np.array([extract_attributes(price) for price in independent_vars[0]])
    print(f"Shape da entrada: {independent_vars.shape}")

    # Padronizar as variáveis independentes
    independent_vars_scaled = scaler_x.transform(independent_vars)

    # Realizar a previsão
    predicted_scaled = loaded_model.predict(independent_vars_scaled)

    # Inverter a padronização da previsão
    predicted_real = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1))

    # Exibir o resultado
    print(f"Previsão do preço: {predicted_real}")

    return predicted_real


def extract_attributes(prediction: PredictPrice) -> List[float]:
    """
    Extrai as características necessárias para a previsão a partir de um objeto PredictPrice.

    Args:
        prediction (PredictPrice): Objeto contendo os atributos quilate, x, y e z.

    Returns:
        List[float]: Lista com os atributos extraídos na ordem correta.
    """
    return [prediction.quilate, prediction.x, prediction.y, prediction.z]
