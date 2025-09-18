import sqlite3

# 连接到数据库，如果文件不存在会自动创建
conn = sqlite3.connect('parking.db')
cursor = conn.cursor()

# 创建表
# 创建停车场基本信息表
# 此表用于存储停车场的基本信息，包括名称、地址、容量和收费标准
cursor.execute('''
CREATE TABLE IF NOT EXISTS parking_lots (
    parking_lot_id INTEGER PRIMARY KEY,           -- 停车场唯一标识符，主键
    name TEXT NOT NULL,                           -- 停车场名称，不能为空
    address TEXT NOT NULL,                        -- 停车场地址，不能为空
    total_spaces INTEGER NOT NULL DEFAULT 0,      -- 总车位数，默认值为0，不能为空
    available_spaces INTEGER NOT NULL DEFAULT 0,  -- 可用车位数，默认值为0，不能为空
    hourly_rate DECIMAL(8,2) NOT NULL             -- 每小时收费标准，精确到分(8位总数，2位小数)
);
''')

# 创建车位详细信息表
# 此表用于存储每个具体车位的信息，包括所属停车场、编号、类型和当前状态
cursor.execute('''
CREATE TABLE IF NOT EXISTS parking_spaces (
    space_id INTEGER PRIMARY KEY,                 -- 车位唯一标识符，主键
    parking_lot_id INTEGER NOT NULL,              -- 所属停车场ID，外键关联parking_lots表
    space_number TEXT NOT NULL,                   -- 车位编号(如"A区001")，不能为空
    space_type TEXT CHECK(space_type IN ('standard', 'disabled', 'VIP', 'electric')) DEFAULT 'standard',
                                                  -- 车位类型：标准、残疾人、VIP、电动车，默认为标准车位
    current_status TEXT CHECK(current_status IN ('available', 'occupied', 'reserved', 'maintenance')) DEFAULT 'available',
                                                  -- 当前状态：可用、占用、预留、维护中，默认为可用
    vehicle_id TEXT DEFAULT NULL,                 -- 当前停放车辆ID，默认为空(无车辆)
    FOREIGN KEY (parking_lot_id) REFERENCES parking_lots (parking_lot_id) ON DELETE CASCADE
                                                  -- 外键约束，确保车位属于已存在的停车场 ON DELETE CASCADE明确外键删除时的行为
);
''')

# 创建停车记录表
# 此表用于记录车辆的进出信息、费用和支付状态
cursor.execute('''
CREATE TABLE parking_records (
    parking_record_id INTEGER PRIMARY KEY AUTOINCREMENT, -- 停车记录唯一标识，主键且自动递增
    vehicle_license TEXT NOT NULL,                   -- 车牌号码，不能为空
    parking_lot_id INTEGER NOT NULL,                 -- 停车场ID，外键关联parking_lots表
    space_id INTEGER NOT NULL,                       -- 使用车位ID，外键关联parking_spaces表
    entry_time DATETIME NOT NULL,                    -- 入场时间，不能为空
    exit_time DATETIME DEFAULT NULL,                 -- 出场时间，默认为空(表示车辆尚未出场)
    total_cost REAL DEFAULT 0.00,                    -- 总费用，默认为0.00
    payment_status TEXT CHECK(payment_status IN ('unpaid', 'paid', 'cancelled')) DEFAULT 'unpaid',
                                                  -- 支付状态：未支付、已支付、已取消，默认为未支付
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 记录创建时间，默认为当前时间戳
    FOREIGN KEY (parking_lot_id) REFERENCES parking_lots(parking_lot_id) ON DELETE CASCADE,
                                                  -- 外键约束，确保记录关联到已存在的停车场 ON DELETE CASCADE明确外键删除时的行为
    FOREIGN KEY (space_id) REFERENCES parking_spaces(space_id) ON DELETE CASCADE
                                                  -- 外键约束，确保记录关联到已存在的车位 ON DELETE CASCADE明确外键删除时的行为
);
''')

# 提交更改
conn.commit()
print("数据库和表已成功创建。")

# 插入停车场数据
cursor.execute("INSERT INTO parking_lots(parking_lot_id, name, address, total_spaces, available_spaces, hourly_rate) VALUES (?, ?, ?, ?, ?, ?)", (1, '停车场1', '位置1', 200, 150, 5.00))  # 停车场ID为1，200个车位，150个可用，每小时5元
cursor.execute("INSERT INTO parking_lots(parking_lot_id, name, address, total_spaces, available_spaces, hourly_rate) VALUES (?, ?, ?, ?, ?, ?)", (2, '停车场2', '位置2', 300, 250, 6.00))  # 停车场ID为2，300个车位，250个可用，每小时6元
cursor.execute("INSERT INTO parking_lots(parking_lot_id, name, address, total_spaces, available_spaces, hourly_rate) VALUES (?, ?, ?, ?, ?, ?)", (3, '停车场3', '位置3', 100, 80, 4.00))  # 停车场ID为3，100个车位，80个可用，每小时4元
conn.commit()
print("插入停车场数据成功。")

# 插入车位信息

cursor.executemany("INSERT INTO parking_spaces (space_id, parking_lot_id, space_number, space_type, current_status, vehicle_id) VALUES (?, ?, ?, ?, ?, ?)",
               [
                    (101, 1, 'A001', 'standard', 'available', None),  # 标准车位，可用
                    (102, 1, 'A002', 'standard', 'available', None),  # 标准车位，可用
                    (103, 1, 'A003', 'standard', 'occupied', '京A12345'),  # 标准车位，已被车牌京A12345占用
                    (104, 1, 'B001', 'VIP', 'available', None),  # VIP车位，可用
                    (105, 1, 'C001', 'disabled', 'available', None),  # 残疾人车位，可用
                    (106, 1, 'D001', 'electric', 'maintenance', None),  # 电动车位，维护中
                    (201, 2, 'P001', 'standard', 'available', None),  # 标准车位，可用
                    (202, 2, 'P002', 'standard', 'available', None),  # 标准车位，可用
                    (203, 2, 'P003', 'VIP', 'reserved', None),  # VIP车位，已预留
                    (204, 2, 'P004', 'electric', 'available', None),  # 电动车位，可用
                    (301, 3, 'L001', 'standard', 'available', None),  # 标准车位，可用
                    (302, 3, 'L002', 'standard', 'occupied', '京B54321'),  # 标准车位，已被车牌京B54321占用
                    (303, 3, 'L003', 'standard', 'available', None)  # 标准车位，可用
               ])
conn.commit()
print("插入车位信息数据成功。")

def insert_parking_records(vehicle_license, parking_lot_id, space_id, entry_time, exit_time = None, total_cost = 0.0, payment_status = 'unpaid'):
    cursor.execute("INSERT INTO parking_records (vehicle_license, parking_lot_id, space_id, entry_time, exit_time , total_cost , payment_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (vehicle_license, parking_lot_id, space_id, entry_time, exit_time , total_cost, payment_status))


insert_parking_records('京A12345', 1, 103, '2023-10-15 09:30:00', '2023-10-15 12:45:00', 16.50, 'paid')
insert_parking_records('京C98765', 2, 201, '2023-10-15 10:15:00', '2023-10-15 11:30:00', 9.00, 'paid')
insert_parking_records('京D55555', 3, 302, '2023-10-15 08:00:00', '2023-10-15 18:00:00', 40.00, 'paid')
insert_parking_records('京E12345', 1, 101, '2023-10-15 14:20:00', None, 0.00, 'unpaid')
insert_parking_records('京F67890', 2, 202, '2023-10-15 13:45:00', None, 0.00, 'unpaid')
insert_parking_records('京G11111', 1, 102, '2023-10-14 15:00:00', '2023-10-14 15:10:00', 0.00, 'cancelled')
conn.commit()
print("插入停车记录数据成功。")

# 所有停车场和它们的所停过车辆
print("\n--- 所有停车场和它们的所停过车辆 ---")
cursor.execute('''
SELECT a.name, b.vehicle_license
FROM parking_lots as a
JOIN parking_records as b ON a.parking_lot_id = b.parking_lot_id;
''')

lots_with_spaces = cursor.fetchall()
for name, vehicle_license in lots_with_spaces:
    print(f"停车场名: {name}, 车牌号: {vehicle_license}")

# 更新车辆出库
print("\n--- 更新车辆出库 ---")
cursor.execute("UPDATE parking_records SET exit_time = ?,total_cost = ?  WHERE vehicle_license = ?", ('2023-10-15 18:00:00', 25.00, '京E12345'))
conn.commit()
print("更新'京E12345'出库。")

# 查询更新后的数据
cursor.execute("SELECT vehicle_license, exit_time, total_cost FROM parking_records WHERE vehicle_license = '京E12345'")
updated_parking = cursor.fetchone()
print(f"更新后的信息: 车辆: {updated_parking[0]}, 出库时间: {updated_parking[1]}, 费用: {updated_parking[2]}")

# 删除车辆信息
print("\n--- 删除车辆信息 ---")
cursor.execute("DELETE FROM parking_records WHERE vehicle_license = ?", ('京G11111',))
conn.commit()
print("车辆 '京G11111' 已被删除。")

# 再次查询车辆信息，验证删除操作
print("\n--- 剩余的车辆信息 ---")
cursor.execute("SELECT vehicle_license FROM parking_records")
remaining_spaces = cursor.fetchall()
for vehicle_license in remaining_spaces:
    print(f"车牌号: {vehicle_license[0]}")

# 关闭连接
conn.close()
print("\n数据库连接已关闭。")

