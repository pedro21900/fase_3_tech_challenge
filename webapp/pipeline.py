import logging
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from fastapi import UploadFile
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)


class IngestionPipeline:
    def __init__(self):
        self.scaler_X = None
        self.scaler_y = None
        self.X_scaled = None
        self.y_scaled = None

    @staticmethod
    def remove_outliers(df: DataFrame, columns: list):
        """Remove outliers de colunas específicas do DataFrame."""
        for col in columns:
            data = np.array(df[col])
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
            df.loc[outliers, col] = np.nan
        df.dropna(how='any', inplace=True)

    def normalize_data(self, diamonds_df: DataFrame):
        """Normaliza os dados e salva os scalers."""
        imputer = SimpleImputer(strategy='mean')
        diamonds_df = pd.DataFrame(imputer.fit_transform(diamonds_df), columns=diamonds_df.columns)

        y = diamonds_df['preco'].values.ravel()
        x = diamonds_df.drop('preco', axis=1)

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_scaled = self.scaler_X.fit_transform(x)
        self.y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        logging.info('Salvando scalers...')
        joblib.dump(self.scaler_X, './resources/ml/scaler_X.pkl')
        joblib.dump(self.scaler_y, './resources/ml/scaler_y.pkl')

        return self.X_scaled, self.y_scaled


class TrainModelPipeline:
    @staticmethod
    def train_model_tree_reg_grid_search(x_train: np.ndarray, y_train: np.ndarray):
        """Treina um modelo usando GridSearchCV."""
        pca = PCA(n_components=3, whiten=True, random_state=42)
        model = DecisionTreeRegressor()

        pipeline = make_pipeline(pca, model)

        grid_search = GridSearchCV(pipeline, {}, n_jobs=-1, verbose=3)
        grid_search.fit(x_train, y_train)

        joblib.dump(grid_search.best_estimator_, './resources/ml/modelo_rf.pkl')
        return grid_search.best_estimator_


class EvaluateModelPipeline:
    @staticmethod
    def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray, scaler_y: StandardScaler):
        """Avalia o modelo e retorna as métricas."""
        y_pred = model.predict(x_test)

        y_pred_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

        mae = mean_absolute_error(y_test_real, y_pred_real)
        mse = mean_squared_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test_real, y_pred_real)

        eval_metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape
        }
        logging.info(f'Avaliação do modelo: {eval_metrics}')
        return eval_metrics


def import_csv_file(file: UploadFile) -> Dict[str, Any]:
    """Processa um arquivo CSV, treina o modelo e retorna as métricas de avaliação."""
    logging.info('Iniciando importação...')

    # Carregar o DataFrame
    diamonds_df = pd.read_csv(file.file)

    # Remover colunas desnecessárias
    columns_to_drop = ['cor', 'clareza', 'profundidade', 'corte', 'lg_topo']
    diamonds_df.drop(columns_to_drop, axis=1, inplace=True)

    pipeline = IngestionPipeline()

    logging.info('Removendo outliers...')
    pipeline.remove_outliers(diamonds_df, diamonds_df.columns)

    logging.info('Normalizando dados...')
    x_scaled, y_scaled = pipeline.normalize_data(diamonds_df)

    logging.info('Dividindo os dados em treino e teste...')
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.3, random_state=0)

    logging.info('Treinando o modelo...')
    trainer = TrainModelPipeline()
    best_model = trainer.train_model_tree_reg_grid_search(x_train, y_train)

    logging.info('Avaliando o modelo...')
    evaluator = EvaluateModelPipeline()
    return evaluator.evaluate_model(best_model, x_test, y_test, pipeline.scaler_y)
