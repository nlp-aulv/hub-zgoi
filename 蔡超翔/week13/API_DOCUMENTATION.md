# 股票助手chatbi接口文档

## 项目概述

该项目是一个股票助手chatbi，可以进行交互对话，股票查询等内容。后端基于FastAPI框架开发，提供RESTful API接口供前端调用。

- 基础url: `http://localhost:8000`

## 接口列表

**总计：21个API接口**

- 用户管理：7个接口
- 聊天功能：6个接口
- 数据管理：4个接口
- 股票收藏：4个接口

*注：本API文档仅包含 `routers` 文件夹中发现的接口。文档中未包含用户提及的“MCP协议接口”和“股票行情API”的额外接口，因为它们在此次整理的文件中未找到详细信息。*

## 1. 用户管理接口

### 1.1 用户登录

- 接口路径：`POST /v1/users/login`
- 描述：实现用户登录功能
- 文件目录：`routers/user.py`
- 函数名：`user_login`
- 请求体：
```json
{
    "user_name": "string", // 用户名
    "password": "string"   // 密码
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": []             // 响应数据
}
```

### 1.2 用户注册

- 接口路径：`POST /v1/users/register`
- 描述：实现用户注册功能
- 文件目录：`routers/user.py`
- 函数名：`user_register`
- 请求体：
```json
{
    "user_name": "string", // 用户名
    "password": "string",  // 密码
    "user_role": "string"  // 用户身份
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": []             // 响应数据
}
```

### 1.3 密码重置

- 接口路径：`POST /v1/users/reset-password`
- 描述：实现用户密码重置功能
- 文件目录：`routers/user.py`
- 函数名：`user_reset_password`
- 请求体：
```json
{
    "user_name": "string",     // 用户名
    "password": "string",      // 旧密码
    "new_password": "string"   // 新密码
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": []             // 响应数据
}
```

### 1.4 获取用户信息

- 接口路径：`POST /v1/users/info`
- 描述：根据用户名获取用户信息
- 文件目录：`routers/user.py`
- 函数名：`user_info`
- 请求体：
```json
{
    "user_name": "string" // 用户名
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": {
        "user_name": "string",
        "user_role": "string",
        "status": "string"
    }             // 响应数据
}
```

### 1.5 重置用户信息

- 接口路径：`POST /v1/users/reset-info`
- 描述：重置用户角色或状态
- 文件目录：`routers/user.py`
- 函数名：`user_reset_info`
- 请求体：
```json
{
    "user_name": "string", // 用户名
    "user_role": "string", // 新用户身份 (可选)
    "status": "string"     // 新状态 (可选)
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": []             // 响应数据
}
```

### 1.6 删除用户

- 接口路径：`POST /v1/users/delete`
- 描述：删除指定用户
- 文件目录：`routers/user.py`
- 函数名：`user_delete`
- 请求体：
```json
{
    "user_name": "string" // 用户名
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": []             // 响应数据
}
```

### 1.7 获取用户列表

- 接口路径：`POST /v1/users/list`
- 描述：获取所有用户列表
- 文件目录：`routers/user.py`
- 函数名：`user_list`
- 请求体：
```json
{}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": [
        {
            "user_name": "string",
            "user_role": "string",
            "status": "string"
        }
    ]             // 响应数据
}
```

## 2. 聊天功能接口

### 2.1 聊天交互

- 接口路径：`POST /v1/chat/`
- 描述：处理聊天交互，以流式传输响应
- 文件目录：`routers/chat.py`
- 函数名：`chat`
- 请求体：
```json
{
    "user_name": "string", // 用户名
    "task": "string",
    "session_id": "string",
    "content": "string",
    "tools": []
}
```
- 响应体：
```json
// 流式响应 (Server-Sent Events)
// 或错误时的 BasicResponse
{
    "code": 500,           // 状态码
    "message": "string",   // 响应消息
    "data": []             // 响应数据
}
```

### 2.2 初始化聊天会话

- 接口路径：`POST /v1/chat/init`
- 描述：初始化新的聊天会话
- 文件目录：`routers/chat.py`
- 函数名：`init_chat`
- 请求体：
```json
{}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": {
        "session_id": "string"
    }             // 响应数据
}
```

### 2.3 获取聊天会话

- 接口路径：`POST /v1/chat/get`
- 描述：检索指定会话ID的聊天会话
- 文件目录：`routers/chat.py`
- 函数名：`get_chat`
- 请求体：
```json
{
    "session_id": "string" // 会话ID
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": {}             // 聊天会话数据
}
```

### 2.4 删除聊天会话

- 接口路径：`POST /v1/chat/delete`
- 描述：删除指定会话ID的聊天会话
- 文件目录：`routers/chat.py`
- 函数名：`delete_chat`
- 请求体：
```json
{
    "session_id": "string" // 会话ID
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": []             // 响应数据
}
```

### 2.5 列出用户聊天记录

- 接口路径：`POST /v1/chat/list`
- 描述：列出指定用户的所有聊天记录
- 文件目录：`routers/chat.py`
- 函数名：`list_chat`
- 请求体：
```json
{
    "user_name": "string" // 用户名
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": []             // 聊天记录列表
}
```

### 2.6 聊天消息反馈

- 接口路径：`POST /v1/chat/feedback`
- 描述：提供聊天消息的反馈
- 文件目录：`routers/chat.py`
- 函数名：`feedback_chat`
- 请求体：
```json
{
    "session_id": "string", // 会话ID
    "message_id": 0,        // 消息ID
    "feedback": true        // 反馈 (true/false)
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": []             // 响应数据
}
```

## 3. 数据管理接口

### 3.1 下载数据

- 接口路径：`POST /v1/data/download`
- 描述：数据下载功能的占位符
- 文件目录：`routers/data.py`
- 函数名：`download_data`
- 请求体：
```json
{}
```
- 响应体：
```json
{}
```

### 3.2 创建数据

- 接口路径：`POST /v1/data/create`
- 描述：数据创建功能的占位符
- 文件目录：`routers/data.py`
- 函数名：`create_data`
- 请求体：
```json
{}
```
- 响应体：
```json
{}
```

### 3.3 上传数据

- 接口路径：`POST /v1/data/upload`
- 描述：数据上传功能的占位符
- 文件目录：`routers/data.py`
- 函数名：`upload_data`
- 请求体：
```json
{}
```
- 响应体：
```json
{}
```

### 3.4 删除数据

- 接口路径：`POST /v1/data/delete`
- 描述：数据删除功能的占位符
- 文件目录：`routers/data.py`
- 函数名：`delete_data`
- 请求体：
```json
{}
```
- 响应体：
```json
{}
```

## 4. 股票收藏接口

### 4.1 列出用户收藏股票

- 接口路径：`POST /v1/stock/list_fav_stock`
- 描述：检索用户所有收藏的股票
- 文件目录：`routers/stock.py`
- 函数名：`get_user_all_stock`
- 请求体：
```json
{
    "user_name": "string" // 用户名
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": [
        "string" // 股票代码列表
    ]             // 响应数据
}
```

### 4.2 删除用户收藏股票

- 接口路径：`POST /v1/stock/del_fav_stock`
- 描述：删除用户的收藏股票
- 文件目录：`routers/stock.py`
- 函数名：`delete_user_stock`
- 请求体：
```json
{
    "user_name": "string", // 用户名
    "stock_code": "string" // 股票代码
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": true           // 操作结果 (true/false)
}
```

### 4.3 添加用户收藏股票

- 接口路径：`POST /v1/stock/add_fav_stock`
- 描述：添加用户的收藏股票
- 文件目录：`routers/stock.py`
- 函数名：`add_user_stock`
- 请求体：
```json
{
    "user_name": "string", // 用户名
    "stock_code": "string" // 股票代码
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": true           // 操作结果 (true/false)
}
```

### 4.4 清空用户收藏股票

- 接口路径：`POST /v1/stock/clear_fav_stock`
- 描述：清除用户所有收藏的股票
- 文件目录：`routers/stock.py`
- 函数名：`clear_user_stock`
- 请求体：
```json
{
    "user_name": "string" // 用户名
}
```
- 响应体：
```json
{
    "code": 200,           // 状态码
    "message": "string",   // 响应消息
    "data": true           // 操作结果 (true/false)
}
```
