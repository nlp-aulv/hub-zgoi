# 数据交互模型
## User（用户）

| 字段名           | 字段类型     | 字段描述 |
| ------------- | -------- | ---- |
| user_id       | int      | 用户ID |
| user_name     | str      | 用户名  |
| user_role     | str      | 用户权限 |
| register_time | datetime | 注册时间 |
| status        | bool     | 用户状态 |
## BasicResponse(基础响应)

| 字段名     | 字段类型                            | 字段描述           |
| ------- | ------------------------------- | -------------- |
| code    | int                             | 状态码            |
| message | str                             | 响应消息           |
| data    | Optional[Union[List[Any], Any]] | 响应数据，支持列表或任意类型 |
## RequestForUserLogin（用户注册请求）
| 字段名       | 字段类型 | 字段描述 |
| --------- | ---- | ---- |
| user_name | str  | 用户名  |
| password  | str  | 密码   |
| user_role | str  | 用户权限 |
## RequestForUserLogin（用户登陆请求）
| 字段名       | 字段类型 | 字段描述 |
| --------- | ---- | ---- |
| user_name | str  | 用户名  |
| password  | str  | 密码   |
## RequestForUserResetPassword（密码重置请求）
| 字段名          | 字段类型 | 字段描述 |
| ------------ | ---- | ---- |
| user_name    | str  | 用户名  |
| password     | str  | 密码   |
| new_password | str  | 新密码  |
## RequestForUserChangeInfo（用户信息修改请求）
| 字段名       | 字段类型           | 字段描述 |
| --------- | -------------- | ---- |
| user_name | str            | 用户名  |
| user_role | Optional[str]  | 用户权限 |
| status    | Optional[bool] | 用户状态 |
## RequestForChat（聊天请求）
| 字段名              | 字段类型                 | 字段描述     |
| ---------------- | -------------------- | -------- |
| content          | str                  | 用户ID     |
| user_name        | str                  | 用户名      |
| session_id       | Optional[str]        | 会话ID     |
| task             | Optional[str]        | 对话任务类型   |
| tools            | Optional[List[str]]  | 工具列表     |
| image_content    | Optional[str]        | 图像内容     |
| file_content     | Optional[str]        | 文件内容     |
| url_content      | Optional[str]        | 链接内容     |
| audio_content    | Optional[str]        | 音频内容     |
| video_content    | Optional[str]        | 视频内容     |
| vison_mode       | Optional[bool]=False | 视觉模式     |
| deepsearch_mode  | Optional[bool]=False | 深度搜索模式   |
| sql_interpreter  | Optional[bool]=False | SQL解释器模式 |
| code_interpreter | Optional[bool]=False | 代码解释器模式  |
## ResponseForChat（聊天响应）
| 字段名           | 字段类型          | 字段描述      |
| ------------- | ------------- | --------- |
| response_text | str           | 主要的文本回复内容 |
| session_id    | Optional[str] | 会话ID      |
| response_code | Optional[str] | 响应代码      |
| response_sql  | Optional[str] | SQL查询响应   |

## StockFavInfo（股票收藏信息）

| 字段名         | 字段类型     | 字段描述 |
| ----------- | -------- | ---- |
| stock_code  | str      | 股票代码 |
| create_time | datetime | 收藏时间 |
## ChatSession（聊天会话信息）
| 字段名           | 字段类型               | 字段描述         |
| ------------- | ------------------ | ------------ |
| user_id       | int                | 用户ID         |
| session_id    | str                | 会话ID         |
| title         | str                | 会话标题         |
| start_time    | datetime           | 会话开始时间       |
| feedback      | Optional[bool]     | 用户反馈（满意/不满意） |
| feedback_time | Optional[datetime] | 反馈时间         |

# FastApi接口

## 1. **聊天相关接口**​ (`/v1/chat`)

**基础路径：**`/v1/chat`

| 方法   | 端点          | 功能         | 请求参数                                             | 响应类型              |
| ---- | ----------- | ---------- | ------------------------------------------------ | ----------------- |
| POST | `/`         | 聊天对话（流式输出） | RequestForChat                                   | StreamingResponse |
| POST | `/init`     | 初始化聊天会话    | 无                                                | BasicResponse     |
| POST | `/get`      | 获取聊天会话详情   | session_id: str                                  | BasicResponse     |
| POST | `/delete`   | 删除聊天会话     | session_id: str                                  | BasicResponse     |
| POST | `/list`     | 列出用户聊天记录   | user_name: str                                   | BasicResponse     |
| POST | `/feedback` | 聊天反馈评分     | session_id: str, message_id: int, feedback: bool | BasicResponse     |

---

## 2. **股票收藏接口**​ (`/v1/stock`)

**基础路径：** `/v1/stock`

|方法|端点|功能|请求参数|响应类型|
|---|---|---|---|---|
|POST|`/list_fav_stock`|获取用户收藏股票|user_name: str|BasicResponse|
|POST|`/del_fav_stock`|删除用户收藏股票|user_name: str, stock_code: str|BasicResponse|
|POST|`/add_fav_stock`|添加用户收藏股票|user_name: str, stock_code: str|BasicResponse|
|POST|`/clear_fav_stock`|清空用户收藏股票|user_name: str|BasicResponse|

---

## 3. **用户管理接口**​ (`/v1/users`)

**基础路径：** `/v1/users`

| 方法   | 端点                | 功能     | 请求参数                          | 响应类型          |
| ---- | ----------------- | ------ | ----------------------------- | ------------- |
| POST | `/login`          | 用户登录   | RequestForUserLogin对象         | BasicResponse |
| POST | `/register`       | 用户注册   | RequestForUserRegister对象      | BasicResponse |
| POST | `/reset-password` | 重置密码   | RequestForUserResetPassword对象 | BasicResponse |
| POST | `/info`           | 获取用户信息 | user_name: str                | BasicResponse |
| POST | `/reset-info`     | 修改用户信息 | RequestForUserChangeInfo对象    | BasicResponse |
| POST | `/delete`         | 删除用户   | user_name: str                | BasicResponse |
| POST | `/list`           | 获取用户列表 | 无                             | BasicResponse |
## 4. 股票数据接口（同时作为MCP服务）

**基础路径：** `https://api.autostock.cn/v1/stock/`

|方法|端点|功能描述|必需参数|可选参数|参数说明|
|---|---|---|---|---|---|
|**GET**​|`/get_stock_code`|所有股票代码查询|无|`keyword`|支持代码和名称模糊查询|
|**GET**​|`/get_index_code`|所有指数代码查询|无|无|获取所有指数代码和名称|
|**GET**​|`/get_industry_code`|板块数据查询|无|无|获取股票行业板块数据|
|**GET**​|`/get_board_info`|大盘数据查询|无|无|获取大盘整体数据|
|**GET**​|`/get_stock_rank`|股票排行查询|`node`|`industryCode`, `pageIndex`, `pageSize`, `sort`, `asc`|`node`: 市场代码{a,b,ash,asz,bsh,bsz}  <br>`sort`: 排序字段  <br>`asc`: 0=降序,1=升序|
|**GET**​|`/get_month_line`|月K线数据查询|`code`|`startDate`, `endDate`, `type`|`type`: 0不复权,1前复权,2后复权|
|**GET**​|`/get_week_line`|周K线数据查询|`code`|`startDate`, `endDate`, `type`|`type`: 0不复权,1前复权,2后复权|
|**GET**​|`/get_day_line`|日K线数据查询|`code`|`startDate`, `endDate`, `type`|`type`: 0不复权,1前复权,2后复权|
|**GET**​|`/get_stock_info`|股票基础信息查询|`code`|无|获取股票基本信息|
|**GET**​|`/get_stock_minute_data`|分时数据查询|`code`|无|获取股票分时交易数据|


# MCP接口

## 1. 新闻服务 (News-MCP-Server)

| 接口名称       | 函数名                    | 描述                | 输入参数 | 返回数据         |
| ---------- | ---------------------- | ----------------- | ---- | ------------ |
| 获取今日新闻     | `get_today_daily_news` | 从外部API获取今日新闻简报列表  | 无    | 今日新闻列表       |
| 获取抖音热点     | `get_douyin_hot_news`  | 从抖音获取热门趋势话题或热点新闻  | 无    | 抖音热点新闻列表     |
| 获取GitHub热点 | `get_github_hot_news`  | 获取GitHub热门趋势仓库/项目 | 无    | GitHub热门项目列表 |
| 获取头条热点     | `get_toutiao_hot_news` | 从头条获取热点新闻标题       | 无    | 头条热点新闻列表     |
| 获取体育新闻     | `get_sports_news`      | 获取电子竞技或一般体育新闻     | 无    | 体育新闻列表       |

## 2. 名言服务 (Saying-MCP-Server)

| 接口名称   | 函数名                           | 描述            | 输入参数 | 返回数据     |
| ------ | ----------------------------- | ------------- | ---- | -------- |
| 获取今日名言 | `get_today_familous_saying`   | 获取随机名言或"一言"引用 | 无    | 名言内容     |
| 获取励志名言 | `get_today_motivation_saying` | 获取励志名言或激励性引用  | 无    | 励志名言内容   |
| 获取工作名言 | `get_today_working_saying`    | 获取工作相关引用或心灵鸡汤 | 无    | 工作相关名言内容 |

## 3. 工具服务 (Tools-MCP-Server)

| 接口名称   | 函数名                  | 描述              | 输入参数                                                            | 返回数据   |
| ------ | -------------------- | --------------- | --------------------------------------------------------------- | ------ |
| 城市天气查询 | `get_city_weather`   | 使用城市拼音获取当前天气数据  | `city_name`: 城市拼音名称                                             | 天气数据   |
| 地址解析   | `get_address_detail` | 解析地址字符串提取详细组成部分 | `address_text`: 城市名称                                            | 地址详细信息 |
| 电话号码查询 | `get_tel_info`       | 获取电话号码的基本信息     | `tel_no`: 电话号码                                                  | 电话号码信息 |
| 景点信息查询 | `get_scenic_info`    | 搜索并获取特定景点信息     | `scenic_name`: 景点名称                                             | 景点信息列表 |
| 花卉信息查询 | `get_flower_info`    | 获取花卉的花语和详细信息    | `flower_name`: 花卉名称                                             | 花卉信息   |
| 货币汇率转换 | `get_rate_transform` | 计算两种货币之间的汇率转换   | `source_coin`: 源货币代码  <br>`aim_coin`: 目标货币代码  <br>`money`: 转换金额 | 转换后的金额 |