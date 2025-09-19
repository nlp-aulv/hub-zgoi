from pydantic import BaseModel, Field
from typing import Optional, List

# 用户模型
class UserBase(BaseModel):
    username: str
    password: str
    email: str
    phone: Optional[str] = None
    status: int = 1  # 1: 启用, 0: 禁用

class UserCreate(UserBase):
    pass

class UserUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    status: Optional[int] = None

class User(UserBase):
    id: int
    create_time: int
    update_time: int

    class Config:
        orm_mode = True  # 允许从 ORM 对象转换

# 用户详情模型
class UserProfileBase(BaseModel):
    user_id: int
    real_name: str
    gender: int = 0  # 0:未知, 1:男, 2:女
    age: Optional[int] = None
    avatar: Optional[str] = None

class UserProfileCreate(UserProfileBase):
    pass

class UserProfileUpdate(BaseModel):
    real_name: Optional[str] = None
    gender: Optional[int] = None
    age: Optional[int] = None
    avatar: Optional[str] = None

class UserProfile(UserProfileBase):
    id: int
    create_time: int
    update_time: int

    class Config:
        orm_mode = True

