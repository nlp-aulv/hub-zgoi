import sqlite3
import time
from pathlib import Path

from 陈宇豪.week06.user import *

DB_PATH = Path("user_db.db")


def init_db():
    """初始化数据库并创建表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 创建用户表
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS user
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       username
                       TEXT
                       NOT
                       NULL
                       UNIQUE,
                       password
                       TEXT
                       NOT
                       NULL,
                       email
                       TEXT
                       NOT
                       NULL
                       UNIQUE,
                       phone
                       TEXT,
                       status
                       INTEGER
                       DEFAULT
                       1,
                       create_time
                       INTEGER
                       NOT
                       NULL,
                       update_time
                       INTEGER
                       NOT
                       NULL
                   )
                   """)

    # 创建用户详情表
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS user_profile
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       user_id
                       INTEGER
                       NOT
                       NULL
                       UNIQUE,
                       real_name
                       TEXT
                       NOT
                       NULL,
                       gender
                       INTEGER
                       DEFAULT
                       0,
                       age
                       INTEGER,
                       avatar
                       TEXT,
                       create_time
                       INTEGER
                       NOT
                       NULL,
                       update_time
                       INTEGER
                       NOT
                       NULL,
                       FOREIGN
                       KEY
                   (
                       user_id
                   ) REFERENCES user
                   (
                       id
                   ) ON DELETE CASCADE
                       )
                   """)

    conn.commit()
    conn.close()


def get_db():
    """获取数据库连接"""
    return sqlite3.connect(DB_PATH)


def create_user(user: UserCreate):
    """创建新用户"""
    conn = get_db()
    cursor = conn.cursor()

    # 生成时间戳（毫秒）
    now = int(time.time() * 1000)

    cursor.execute("""
                   INSERT INTO user (username, password, email, phone, status, create_time, update_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   """, (
                       user.username,
                       user.password,
                       user.email,
                       user.phone,
                       user.status,
                       now,
                       now
                   ))

    user_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return user_id


def get_user(user_id: int) -> User:
    """获取用户详情"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
                   SELECT id,
                          username,
                          password,
                          email,
                          phone,
                          status,
                          create_time,
                          update_time
                   FROM user
                   WHERE id = ?
                   """, (user_id,))

    row = cursor.fetchone()
    if not row:
        raise ValueError("User not found")

    # 转换为User模型
    user = User(
        id=row[0],
        username=row[1],
        password=row[2],
        email=row[3],
        phone=row[4],
        status=row[5],
        create_time=row[6],
        update_time=row[7]
    )
    conn.close()
    return user


def update_user(user_id: int, user_update: UserUpdate):
    """更新用户信息"""
    conn = get_db()
    cursor = conn.cursor()

    now = int(time.time() * 1000)
    updates = []
    params = []

    # 构建更新字段
    if user_update.username is not None:
        updates.append("username = ?")
        params.append(user_update.username)
    if user_update.password is not None:
        updates.append("password = ?")
        params.append(user_update.password)
    if user_update.email is not None:
        updates.append("email = ?")
        params.append(user_update.email)
    if user_update.phone is not None:
        updates.append("phone = ?")
        params.append(user_update.phone)
    if user_update.status is not None:
        updates.append("status = ?")
        params.append(user_update.status)

    # 添加更新时间
    updates.append("update_time = ?")
    params.append(now)

    # 执行更新
    cursor.execute(
        f"UPDATE user SET {', '.join(updates)} WHERE id = ?",
        (*params, user_id)
    )

    conn.commit()
    conn.close()


def create_user_profile(user_id: int, profile: UserProfileCreate):
    """创建用户详情"""
    conn = get_db()
    cursor = conn.cursor()

    now = int(time.time() * 1000)

    cursor.execute("""
                   INSERT INTO user_profile (user_id, real_name, gender, age, avatar, create_time, update_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   """, (
                       user_id,
                       profile.real_name,
                       profile.gender,
                       profile.age,
                       profile.avatar,
                       now,
                       now
                   ))

    conn.commit()
    conn.close()


def get_user_profile(user_id: int) -> UserProfile:
    """获取用户详情"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
                   SELECT id,
                          user_id,
                          real_name,
                          gender,
                          age,
                          avatar,
                          create_time,
                          update_time
                   FROM user_profile
                   WHERE user_id = ?
                   """, (user_id,))

    row = cursor.fetchone()
    if not row:
        raise ValueError("Profile not found")

    profile = UserProfile(
        id=row[0],
        user_id=row[1],
        real_name=row[2],
        gender=row[3],
        age=row[4],
        avatar=row[5],
        create_time=row[6],
        update_time=row[7]
    )
    conn.close()
    return profile