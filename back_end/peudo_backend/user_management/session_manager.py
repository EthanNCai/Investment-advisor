"""
会话管理模块，处理用户登录会话
"""
import time
import random
import string
from typing import Dict, Any, Optional

# 会话存储 {session_id: {user_info, expire_time}}
SESSION_STORE: Dict[str, Dict[str, Any]] = {}

# 会话过期时间（秒）
SESSION_EXPIRE_SECONDS = 24 * 3600  # 24小时


def create_session(user_info: Dict[str, Any]) -> str:
    """
    创建新会话
    
    参数:
        user_info: 用户信息
        
    返回:
        会话ID
    """
    # 生成唯一会话ID
    session_id = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(32))

    # 存储会话信息
    SESSION_STORE[session_id] = {
        'user': user_info,
        'expire_time': time.time() + SESSION_EXPIRE_SECONDS
    }

    # 清理过期会话
    clean_expired_sessions()

    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    获取会话信息
    
    参数:
        session_id: 会话ID
        
    返回:
        会话信息，如果会话不存在或已过期则返回None
    """
    if session_id not in SESSION_STORE:
        return None

    session_info = SESSION_STORE[session_id]

    # 检查会话是否过期
    if time.time() > session_info['expire_time']:
        del SESSION_STORE[session_id]
        return None

    # 刷新会话过期时间
    session_info['expire_time'] = time.time() + SESSION_EXPIRE_SECONDS

    return session_info['user']


def update_session(session_id: str, user_info: Dict[str, Any]) -> bool:
    """
    更新会话信息
    
    参数:
        session_id: 会话ID
        user_info: 新的用户信息
        
    返回:
        更新是否成功
    """
    if session_id not in SESSION_STORE:
        return False

    SESSION_STORE[session_id]['user'] = user_info
    return True


def delete_session(session_id: str) -> bool:
    """
    删除会话
    
    参数:
        session_id: 会话ID
        
    返回:
        删除是否成功
    """
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]
        return True
    return False


def clean_expired_sessions() -> None:
    """清理过期会话"""
    current_time = time.time()
    expired_sessions = [
        session_id for session_id, session_info in SESSION_STORE.items()
        if current_time > session_info['expire_time']
    ]

    for session_id in expired_sessions:
        del SESSION_STORE[session_id]
