import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class WeightedEnsemble:
    def __init__(self) -> None:
        self.models = ['Catboost', 'XGB', 'ft_transformer', 'LGBM']

    def apply_weights(self, df: pd.DataFrame, weight_type: str = 'mean') -> pd.DataFrame:
        """모델 예측값에 가중치를 적용하여 최종 예측값 계산

        Args:
            df (pd.DataFrame): 모델 예측값과 기타 데이터를 포함하는 데이터프레임.
            weight_type (str): 적용할 가중치 유형, 'mean' 또는 'std' 중 선택.

        Returns:
            pd.DataFrame: 최종 예측값 열이 추가된 데이터프레임을 반환합니다.
        """
        if weight_type == 'mean':
            value_by_day = df.groupby('contract_year_month_day')[self.models].mean()
        elif weight_type == 'std':
            value_by_day = df.groupby('contract_year_month_day')[self.models].std()

        weights = value_by_day.div(value_by_day.sum(axis=1), axis=0).fillna(0)

        weights_expanded = weights.reindex(df['contract_year_month_day']).values
        r_values_expanded = df[self.models].values
        df['final_prediction'] = np.einsum('ij,ij->i', r_values_expanded, weights_expanded)

        return df

    def visualize(self, val: pd.DataFrame, show_final_pred: bool = False) -> None:
        """모델 예측값과 최종 예측값(옵션)을 시각화

        Args:
            val (pd.DataFrame): 각 모델별 예측값과 최종 예측값(옵션 포함)을 담은 데이터프레임.
            show_final_pred (bool): True일 경우 최종 예측값도 그래프에 포함합니다.
        """
        plt.figure(figsize=(10, 6))
        for model in self.models:
            plt.plot(val.index, val[model], label=model)

        if show_final_pred:
            plt.plot(val.index, val['final_prediction'], label='Final Prediction', linewidth=2, color='black')

        plt.title('Average of Model Predictions')
        plt.ylabel('Average Value')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.show()
