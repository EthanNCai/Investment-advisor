"""
预测模块初始化文件
提供时间序列预测模型和工具
"""

from .lstm_predictor import LSTMPredictor
from .enhanced_lstm_predictor import EnhancedLSTMPredictor
from .prediction_utils import preprocess_data, postprocess_results, evaluate_model_performance


# 根据模型类型选择预测器
def get_predictor(model_type="lstm", **kwargs):
    """
    根据指定的模型类型返回相应的预测器实例
    
    参数:
        model_type: 模型类型，可选值为'lstm'、'enhanced_lstm'
        **kwargs: 传递给预测器构造函数的其他参数
        
    返回:
        预测器实例
    """
    if model_type == "enhanced_lstm":
        return EnhancedLSTMPredictor(**kwargs)
    else:
        return LSTMPredictor(**kwargs)
