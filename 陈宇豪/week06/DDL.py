# 创建表
import sqlite3

# 连接到数据库，如果文件不存在会自动创建
conn = sqlite3.connect('library.db')
cursor = conn.cursor()

# 创建 -- 用户表
cursor.execute('''
               CREATE TABLE `user`
               (
                   `id`          BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                   `username`    VARCHAR(50)  NOT NULL UNIQUE,
                   `password`    VARCHAR(100) NOT NULL ,
                   `email`       VARCHAR(100) NOT NULL UNIQUE,
                   `phone`       VARCHAR(20) DEFAULT NULL,
                   `status`      TINYINT     DEFAULT 1 ,
                   `create_time` BIGINT       NOT NULL,
                   `update_time` BIGINT       NOT NULL
               );
               ''')

# 创建 -- 用户表
cursor.execute('''
               CREATE TABLE `user_profile`
               (
                   `id`          BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                   `user_id`     BIGINT UNSIGNED NOT NULL ,
                   `real_name`   VARCHAR(50)  DEFAULT NULL,
                   `gender`      TINYINT      DEFAULT 0 ,
                   `age`         INT          DEFAULT NULL,
                   `avatar`      VARCHAR(255) DEFAULT NULL ,
                   `address`     VARCHAR(255) DEFAULT NULL,
                   `bio`         TEXT COMMENT '个人简介',
                   `create_time` BIGINT NOT NULL,
                   `update_time` BIGINT NOT NULL);
               ''')

# 创建 -- 角色表
cursor.execute('''
               CREATE TABLE `role`
               (
                   `id`          BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                   `role_name`   VARCHAR(50) NOT NULL UNIQUE,
                   `description` VARCHAR(200) DEFAULT NULL,
                   `status`      TINYINT      DEFAULT 1,
                   `create_time` BIGINT      NOT NULL);
               ''')


#角色关联表
cursor.execute("""
CREATE TABLE `user_role` (
  `id` BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `user_id` BIGINT UNSIGNED NOT NULL,
  `role_id` BIGINT UNSIGNED NOT NULL,
  `create_time` BIGINT NOT NULL
);
""")

# 用户登录日志表
cursor.execute('''
CREATE TABLE `user_login_log` (
  `id` BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `user_id` BIGINT UNSIGNED NOT NULL,
  `login_ip` VARCHAR(45) NOT NULL,
  `user_agent` TEXT ,
  `login_status` TINYINT NOT NULL,
  `fail_reason` VARCHAR(100) DEFAULT NULL,
  `login_time` BIGINT NOT NULL
);
''')



