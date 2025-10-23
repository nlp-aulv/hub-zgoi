import sqlite3

# 连接到数据库，如果文件不存在会自动创建
conn = sqlite3.connect('product.db')
cursor = conn.cursor()

# 创建 prod_name 表

cursor.execute('''
DROP TABLE IF EXISTS ship_info;
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS ship_info (
    did     TEXT NOT NULL,
    bom_id      INTEGER PRIMARY KEY,
    ship_date  TEXT
    
);
''')

cursor.execute('''
DROP TABLE IF EXISTS id_upid_mapping;
''')
# 创建 id_mapping 表，外键关联 ship_info 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS id_upid_mapping (
    did TEXT NOT NULL,
    upid TEXT NOT NULL,
    last_usage_dev_time TEXT,
    FOREIGN KEY (did) REFERENCES ship_info (did)
);
''')

# 创建 dim 表，外键关联 id_upid_mapping 表
cursor.execute('''
DROP TABLE IF EXISTS dim_prod;
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS dim_prod (
    did TEXT NOT NULL,
    prod_name TEXT NOT NULL,
    device_name TEXT,
    FOREIGN KEY (did) REFERENCES id_upid_mapping (did)
);
''')

# 提交更改
conn.commit()
print("数据库和表已成功创建。")


# 插入作者数据
cursor.execute("INSERT INTO ship_info (did, ship_date) VALUES (?,?)", ('123','20250101'))
cursor.execute("INSERT INTO ship_info (did, ship_date) VALUES (?,?)", ('124', '20250101'))
cursor.execute("INSERT INTO ship_info (did, ship_date) VALUES (?,?)", ('567', '20250121'))
conn.commit()
print("数据成功插入。")

# 插入书籍数据
# J.K. Rowling 的 author_id 可能是 1，我们用 SELECT 查询来获取
cursor.execute("SELECT bom_id FROM ship_info WHERE did = '123'")
bom_id = cursor.fetchone()[0]
print('bom_id',bom_id)
print('')

cursor.execute("INSERT INTO id_upid_mapping (did, upid, last_usage_dev_time) VALUES (?, ?, ?)", ('123', 'uer1', '20250501'))
cursor.execute("INSERT INTO id_upid_mapping (did, upid, last_usage_dev_time) VALUES (?, ?, ?)", ('124', 'uer2', '20250601'))


cursor.execute("SELECT did FROM ship_info WHERE did = '123'")
did = cursor.fetchone()[0]
cursor.execute("INSERT INTO dim_prod (did, prod_name, device_name) VALUES (?, ?, ?)", ('123', 'mate10', 'ADF-DD'))
cursor.execute("INSERT INTO dim_prod (did, prod_name, device_name) VALUES (?, ?, ?)", ('124', 'mate20', 'QDF-DD'))

conn.commit()

print("数据已成功插入。")


print("\n--- 发货数据 维表---")
cursor.execute('''
SELECT ship_info.did, ship_info.bom_id,ship_info.ship_date, dim_prod.prod_name, dim_prod.device_name
FROM ship_info
JOIN dim_prod 
ON ship_info.did = dim_prod.did;
''')

books_with_authors = cursor.fetchall()
for did, bom_id,ship_date,prod_name,device_name  in books_with_authors:
    print(f"did: {did}, bom_id: {bom_id},ship_date: {ship_date},, prod_name: {prod_name},device_name: {device_name}")

# 更新产品名称
print("\n--- 产品名称 ---")
cursor.execute("UPDATE dim_prod SET device_name = ? WHERE did = ?", ('567','ASE-QQ'))
conn.commit()
print("产品名称 '123' 的产品信息已更新。")

# 查询更新后的数据
cursor.execute("SELECT did,prod_name, device_name FROM dim_prod WHERE did = '124'")
updated_did_dim = cursor.fetchone()
print(f"更新后的信息: did: {updated_did_dim[0]}, prod_name: {updated_did_dim[1]}, device_name: {updated_did_dim[2]}")

# 删除一个
print("\n--- 删除产品 ---")
cursor.execute("DELETE FROM dim_prod WHERE did = ?", ('124',))
conn.commit()
print("124已被删除。")

# 再次查询借阅人列表，验证删除操作
print("\n--- 剩余的产品 ---")
cursor.execute("SELECT did,prod_name,device_name FROM dim_prod")
new_dim_prod = cursor.fetchall()
for did,prod_name,device_name in new_dim_prod:
    print(f"did: {did}, prod_name: {prod_name},device_name: {device_name}")

# 关闭连接
conn.close()
print("\n数据库连接已关闭。")
