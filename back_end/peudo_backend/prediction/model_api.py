"""
模型预测API接口
提供API接口函数用于在FastAPI中调用
支持单独模型和通用模型预测
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

from .model_predictor import get_predictor, StockRatioPredictor
from .universal_model_predictor import predict_with_universal_model

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 默认模型目录
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "trained_models"


def predict_ratio(
    code_a: str,
    code_b: str,
    prediction_days: int = 30,
    confidence_level: float = 0.95,
    model_name: str = None,
    use_universal_model: bool = True  # 默认使用通用模型
) -> Dict[str, Any]:
    """
    预测两只股票的价格比值
    
    参数:
        code_a: 股票A代码
        code_b: 股票B代码
        prediction_days: 预测天数
        confidence_level: 置信水平 (不使用，保持兼容性)
        model_name: 指定使用的模型名称，如果为None则使用最新模型
        use_universal_model: 是否使用通用模型，True表示使用通用模型，False表示使用专用模型
        
    返回:
        预测结果字典
    """
    start_time = time.time()
    
    # 优先尝试使用通用模型
    if use_universal_model:
        try:
            # 使用通用模型进行预测
            result = predict_with_universal_model(
                code_a=code_a,
                code_b=code_b,
                prediction_days=prediction_days,
                model_name=model_name if model_name and "universal" in model_name else None
            )
            
            # 如果成功预测，直接返回结果
            if "error" not in result:
                # 添加数据源标记
                result["model_type"] = "universal"
                return result
                
            # 如果通用模型预测失败，记录错误并尝试使用专用模型
            logger.warning(f"通用模型预测失败: {result['error']}，尝试使用专用模型")
            
        except Exception as e:
            logger.error(f"通用模型预测发生错误: {e}")
            # 继续使用专用模型
    
    try:
        # 获取预测器实例
        predictor = get_predictor(
            model_dir=str(DEFAULT_MODEL_DIR),
            default_model=model_name
        )
        
        # 检查模型是否已加载
        if not predictor.model_loaded:
            if model_name:
                # 尝试加载指定模型
                success = predictor.load_model(model_name)
                if not success:
                    return {
                        "error": f"无法加载指定模型: {model_name}",
                        "elapsed_time": time.time() - start_time
                    }
            else:
                # 尝试加载最新模型
                success = predictor._load_latest_model()
                if not success:
                    return {
                        "error": "无法加载模型，请先训练模型",
                        "elapsed_time": time.time() - start_time
                    }
        
        # 执行预测
        result = predictor.predict(
            stock_code_a=code_a,
            stock_code_b=code_b,
            days_to_predict=prediction_days
        )
        
        # 如果预测失败，返回错误信息
        if "error" in result:
            return {
                "error": result["error"],
                "elapsed_time": time.time() - start_time
            }
        
        # 添加API兼容性字段
        result["elapsed_time"] = time.time() - start_time
        
        # 重命名一些字段以保持与现有API的兼容性
        result["prediction_dates"] = result.pop("dates")
        result["prediction_values"] = result.pop("values")
        
        # 添加数据源标记
        result["model_type"] = "specialized"
        
        return result
    
    except Exception as e:
        logger.error(f"预测过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "error": f"预测失败: {str(e)}",
            "elapsed_time": time.time() - start_time
        }


def get_available_models() -> List[Dict[str, Any]]:
    """
    获取可用的预训练模型列表
    
    返回:
        模型信息列表
    """
    models = []
    
    try:
        # 检查模型目录是否存在
        if not DEFAULT_MODEL_DIR.exists():
            return []
            
        # 查找所有模型目录
        model_dirs = [d for d in DEFAULT_MODEL_DIR.glob("*") if d.is_dir()]
        
        # 按修改时间排序
        model_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # 提取模型信息
        for model_dir in model_dirs:
            # 读取元数据
            metadata_file = model_dir / "metadata.json"
            metadata = {}
            
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    try:
                        metadata = json.load(f)
                    except:
                        pass
            
            # 获取模型最后修改时间
            last_modified = datetime.fromtimestamp(
                os.path.getmtime(model_dir)
            ).strftime('%Y-%m-%d %H:%M:%S')
            
            # 解析模型名称以获取信息
            model_name = model_dir.name
            
            # 判断是通用模型还是专用模型
            is_universal = model_name.startswith("universal_stock_model")
            
            if is_universal:
                # 通用模型
                model_info = {
                    "name": model_name,
                    "path": str(model_dir),
                    "stock_pair": "通用模型",
                    "model_type": "universal",
                    "last_modified": last_modified,
                    "created_at": metadata.get("created_at", "未知"),
                    "session_id": metadata.get("session_id", ""),
                    "n_stocks": metadata.get("n_stocks", 0),
                    "forecast_horizon": metadata.get("forecast_horizon", 5)
                }
            else:
                # 专用模型
                stock_pair = "未知"
                model_type = "未知"
                
                parts = model_name.split("_")
                if len(parts) >= 3:
                    stock_pair = f"{parts[0]}/{parts[1]}"
                    model_type = parts[2]
                
                # 构建模型信息
                model_info = {
                    "name": model_name,
                    "path": str(model_dir),
                    "stock_pair": stock_pair,
                    "model_type": model_type,
                    "last_modified": last_modified,
                    "created_at": metadata.get("created_at", "未知"),
                    "session_id": metadata.get("session_id", ""),
                }
            
            models.append(model_info)
        
        return models
    
    except Exception as e:
        logger.error(f"获取模型列表时出错: {e}")
        return []


def delete_model(model_name: str) -> Dict[str, Any]:
    """
    删除指定的预训练模型
    
    参数:
        model_name: 模型名称
        
    返回:
        操作结果
    """
    try:
        model_path = DEFAULT_MODEL_DIR / model_name
        
        if not model_path.exists():
            return {"success": False, "message": f"模型不存在: {model_name}"}
        
        # 检查是否为目录
        if not model_path.is_dir():
            return {"success": False, "message": f"无效的模型路径: {model_path}"}
            
        # 删除目录及其内容
        import shutil
        shutil.rmtree(model_path)
        
        return {"success": True, "message": f"已删除模型: {model_name}"}
        
    except Exception as e:
        logger.error(f"删除模型时出错: {e}")
        return {"success": False, "message": f"删除失败: {str(e)}"}


if __name__ == "__main__":
    # 测试API
    print("可用模型列表:")
    models = get_available_models()
    for model in models:
        print(f"- {model['name']} ({model['stock_pair']}, {model['model_type']})")
    
    print("\n测试预测 (通用模型):")
    result = predict_ratio('HSI', 'IXIC', prediction_days=5, use_universal_model=True)
    
    if "error" in result:
        print(f"预测失败: {result['error']}")
    else:
        print(f"预测成功，耗时: {result['elapsed_time']:.2f}秒")
        print(f"使用模型类型: {result.get('model_type', '未知')}")
        print(f"预测天数: {len(result['prediction_dates'])}")
        print(f"预测趋势: {result['forecast_trend']}")
        print(f"风险级别: {result['risk_level']}")
        
    print("\n测试预测 (专用模型):")
    result = predict_ratio('HSI', 'IXIC', prediction_days=5, use_universal_model=False)
    
    if "error" in result:
        print(f"预测失败: {result['error']}")
    else:
        print(f"预测成功，耗时: {result['elapsed_time']:.2f}秒")
        print(f"使用模型类型: {result.get('model_type', '未知')}")
        print(f"预测天数: {len(result['prediction_dates'])}")
        print(f"预测趋势: {result['forecast_trend']}")
        print(f"风险级别: {result['risk_level']}") 