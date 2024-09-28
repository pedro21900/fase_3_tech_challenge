from typing import Optional
from pydantic import BaseModel


class PredictPrice(BaseModel):
    """
    Modelo de dados utilizado para prever o preço de itens com base em características geométricas.

    Atributos:
        quilate (Optional[float]): O peso do item em quilates (padrão: None).
        x (Optional[float]): O comprimento do item (padrão: None).
        y (Optional[float]): A largura do item (padrão: None).
        z (Optional[float]): A altura do item (padrão: None).
    """
    quilate: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
