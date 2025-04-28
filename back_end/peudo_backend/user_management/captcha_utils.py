"""
验证码生成和验证工具
"""
import random
import string
import time
from typing import Dict, Tuple

# 存储验证码的字典 {验证码ID: (验证码, 创建时间)}
CAPTCHA_STORE: Dict[str, Tuple[str, float]] = {}

# 验证码过期时间（秒）
CAPTCHA_EXPIRE_SECONDS = 300

def generate_captcha(length: int = 4) -> Tuple[str, str]:
    """
    生成验证码
    
    参数:
        length: 验证码长度
        
    返回:
        (captcha_id, captcha_text): 验证码ID和验证码文本
    """
    # 生成随机验证码（字母和数字的组合）
    characters = string.ascii_uppercase + string.digits
    captcha_text = ''.join(random.choice(characters) for _ in range(length))
    
    # 生成唯一ID
    captcha_id = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(16))
    
    # 存储验证码
    CAPTCHA_STORE[captcha_id] = (captcha_text, time.time())
    
    # 清理过期验证码
    clean_expired_captchas()
    
    return captcha_id, captcha_text

def verify_captcha(captcha_id: str, user_input: str) -> bool:
    """
    验证用户输入的验证码
    
    参数:
        captcha_id: 验证码ID
        user_input: 用户输入的验证码
        
    返回:
        验证是否成功
    """
    if captcha_id not in CAPTCHA_STORE:
        return False
    
    captcha_text, created_time = CAPTCHA_STORE[captcha_id]
    
    # 检查是否过期
    if time.time() - created_time > CAPTCHA_EXPIRE_SECONDS:
        del CAPTCHA_STORE[captcha_id]
        return False
    
    # 验证后删除，确保一次性使用
    del CAPTCHA_STORE[captcha_id]
    
    # 不区分大小写
    return user_input.upper() == captcha_text.upper()

def clean_expired_captchas() -> None:
    """清理过期验证码"""
    current_time = time.time()
    expired_ids = [
        captcha_id for captcha_id, (_, created_time) in CAPTCHA_STORE.items() 
        if current_time - created_time > CAPTCHA_EXPIRE_SECONDS
    ]
    
    for captcha_id in expired_ids:
        del CAPTCHA_STORE[captcha_id] 