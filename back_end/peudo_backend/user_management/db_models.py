"""
用户管理相关的数据库模型
"""
import os
import sqlite3
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

# 数据库路径 - 修正为指向get_stock_data目录下的数据库文件
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "get_stock_data", "stock_kline_data.db")


def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_tables():
    """初始化用户相关表"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 用户表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT UNIQUE,
        created_at TEXT NOT NULL,
        last_login TEXT
    )
    ''')

    # 用户收藏资产表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_favorites (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        stock_code TEXT NOT NULL,
        stock_name TEXT NOT NULL,
        stock_type TEXT NOT NULL,
        added_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id),
        UNIQUE(user_id, stock_code)
    )
    ''')

    # 用户最近查看的资产对表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_recent_pairs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        code_a TEXT NOT NULL,
        name_a TEXT NOT NULL,
        code_b TEXT NOT NULL,
        name_b TEXT NOT NULL,
        viewed_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    """对密码进行哈希处理"""
    return hashlib.sha256(password.encode()).hexdigest()


class UserModel:
    @staticmethod
    def create_user(username: str, password: str, email: str = None) -> Optional[int]:
        """创建新用户"""
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                "INSERT INTO users (username, password_hash, email, created_at) VALUES (?, ?, ?, ?)",
                (username, hash_password(password), email, now)
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # 用户名或邮箱已存在
            return None
        finally:
            conn.close()

    @staticmethod
    def verify_user(username: str, password: str) -> Optional[Dict[str, Any]]:
        """验证用户凭据"""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        )
        user = cursor.fetchone()
        conn.close()

        if user and user['password_hash'] == hash_password(password):
            # 更新最后登录时间
            UserModel.update_last_login(user['id'])
            return dict(user)
        return None

    @staticmethod
    def update_last_login(user_id: int) -> None:
        """更新用户最后登录时间"""
        conn = get_db_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (now, user_id)
        )
        conn.commit()
        conn.close()

    @staticmethod
    def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取用户信息"""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        conn.close()
        return dict(user) if user else None


class FavoriteModel:
    @staticmethod
    def add_favorite(user_id: int, stock_code: str, stock_name: str, stock_type: str) -> bool:
        """添加收藏资产"""
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                "INSERT INTO user_favorites (user_id, stock_code, stock_name, stock_type, added_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, stock_code, stock_name, stock_type, now)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # 已经收藏
            return False
        finally:
            conn.close()

    @staticmethod
    def remove_favorite(user_id: int, stock_code: str) -> bool:
        """删除收藏资产"""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM user_favorites WHERE user_id = ? AND stock_code = ?",
            (user_id, stock_code)
        )
        affected = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return affected

    @staticmethod
    def get_user_favorites(user_id: int) -> List[Dict[str, Any]]:
        """获取用户收藏的资产列表"""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM user_favorites WHERE user_id = ? ORDER BY added_at DESC",
            (user_id,)
        )
        favorites = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return favorites

    @staticmethod
    def is_favorite(user_id: int, stock_code: str) -> bool:
        """检查资产是否已收藏"""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM user_favorites WHERE user_id = ? AND stock_code = ?",
            (user_id, stock_code)
        )
        result = cursor.fetchone() is not None
        conn.close()
        return result


class RecentPairModel:
    @staticmethod
    def add_recent_pair(user_id: int, code_a: str, name_a: str, code_b: str, name_b: str) -> None:
        """添加或更新最近查看的资产对"""
        conn = get_db_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 检查是否已存在相同的资产对（不考虑顺序）
        cursor.execute(
            """
            SELECT id FROM user_recent_pairs 
            WHERE user_id = ? AND 
            ((code_a = ? AND code_b = ?) OR (code_a = ? AND code_b = ?))
            """,
            (user_id, code_a, code_b, code_b, code_a)
        )
        existing = cursor.fetchone()

        if existing:
            # 更新现有记录的时间戳
            cursor.execute(
                "UPDATE user_recent_pairs SET viewed_at = ? WHERE id = ?",
                (now, existing['id'])
            )
        else:
            # 添加新记录
            cursor.execute(
                "INSERT INTO user_recent_pairs (user_id, code_a, name_a, code_b, name_b, viewed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, code_a, name_a, code_b, name_b, now)
            )

            # 检查用户的记录数量
            cursor.execute(
                "SELECT COUNT(*) as count FROM user_recent_pairs WHERE user_id = ?",
                (user_id,)
            )
            count = cursor.fetchone()['count']

            # 如果超过10条，删除最旧的记录
            if count > 10:
                cursor.execute(
                    """
                    DELETE FROM user_recent_pairs 
                    WHERE id IN (
                        SELECT id FROM user_recent_pairs 
                        WHERE user_id = ? 
                        ORDER BY viewed_at ASC 
                        LIMIT ?)
                    """,
                    (user_id, count - 10)
                )

        conn.commit()
        conn.close()

    @staticmethod
    def get_user_recent_pairs(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """获取用户最近查看的资产对列表"""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM user_recent_pairs 
            WHERE user_id = ? 
            ORDER BY viewed_at DESC 
            LIMIT ?
            """,
            (user_id, limit)
        )
        pairs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return pairs


#确保表已创建
init_user_tables()
