已有接口文档：

一、**对话接口**

+ /v1/chat/
  
  + **方法**: POST
  
  + **路径**: `/v1/chat/`
  
  + **功能**: 处理用户聊天请求，返回流式响应
  
  + **特点**: 使用 Server-Sent Events (SSE) 实现实时流式输出

+ /v1/chat/init
  
  - **方法**: POST
  
  - **路径**: `/v1/chat/init`
  
  - **功能**: 创建新的聊天会话，返回 session_id

- /v1/chat/get
  
  - **方法**: POST
  
  - **路径**: `/v1/chat/get`
  
  - **参数**: `session_id` (查询参数)
  
  - **功能**: 根据 session_id 获取历史聊天记录

- /v1/chat/delete
  
  - **方法**: POST
  
  -  **路径**: `/v1/chat/delete`
  
  - **参数**: `session_id` (请求体)
  
  - **功能**: 删除指定的聊天会话

- /v1/chat/feedback
  
  - **方法**: POST
  
  - **路径**: `/v1/chat/feedback`
  
  - **参数**: `session_id`, `message_id`, `feedback`
  
  - **功能**: 用户对某条消息进行好评/差评

二、**数据接口**

+ /v1/data/download
  
  + **功能**: 数据下载
  
  - **方法**: POST
  
  - **路径**: `/v1/data/download`

+ /v1/data/create
  
  - **功能**: 数据创建
  
  - **方法**: POST
  
  - **路径**: `/v1/data/create`

+ /v1/data/upload
  
  + **功能**: 数据上传
  
  - **方法**: POST
  
  - **路径**: `/v1/data/upload`

+ /v1/data/delete
  
  + **功能**: 数据删除
  
  - **方法**: POST
  
  - **路径**: `/v1/data/delete`

三、**stock接口**

+ /v1/stock/list_fav_stock
  
  - **方法**: POST
  
  - **路径**: `/v1/stock/list_fav_stock`
  
  - **功能**: 获取指定用户的所有收藏股票

+ /v1/stock/del_fav_stock
  
  - **方法**: POST
  
  - **路径**: `/v1/stock/del_fav_stock`
  
  - **功能**: 删除用户指定的收藏股票
  
  - **参数**: `user_name`(用户名), `stock_code`(股票代码)

+ /v1/stock/add_fav_stock
  
  - **方法**: POST
  
  - **路径**: `/v1/stock/add_fav_stock`
  
  - **功能**: 为用户添加股票收藏
  
  - **参数**: `user_name`(用户名), `stock_code`(股票代码)

+ /v1/stock/clear_fav_stock
  
  - **方法**: POST
  
  - **路径**: `/v1/stock/clear_fav_stock`
  
  - **功能**: 清空用户的所有股票收藏
  
  - **参数**: `user_name`(用户名)

四、用户接口

+ /v1/users/login
  
  + **方法**：post
  
  + **参数**：RequestForUserLogin
  
  + **功能**：用户登录

+ /v1/users/register
  
  - **方法**：post
  
  - **参数**：RequestForUserLogin
  
  - **功能**：用户注册

+ /v1/users/reset-password
  
  + **方法**：post
  - **参数**：RequestForUserResetPassword
  
  - **功能**：用户重置密码

+ /v1/users/info
  
  - **方法**：post
  
  - **参数**：user_name
  
  - **功能**：用户信息

+ /v1/users/reset-info
  
  - **方法**：post
  
  - **参数**：RequestForUserChangeInfo
  
  - **功能**：用户信息修改

+ /v1/users/delete
  
  - **方法**：post
  
  - **参数**：user_name
  
  - **功能**：用户删除

+ /v1/users/list
  
  - **方法**：post
  
  - **功能**：用户列表
