# 导入 SQLAlchemy 所需的模块
from sqlalchemy import create_engine, Column, Integer, String, Numeric, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.sql import func
from datetime import datetime

# 创建数据库引擎，这里使用 SQLite
# check_same_thread=False 允许在多线程环境下使用，但对于单文件示例可以忽略
engine = create_engine('sqlite:///parking_orm.db', echo=True)

# 创建 ORM 模型的基类
Base = declarative_base()

# --- 定义 ORM 模型（与数据库表对应） ---

# 定义停车场表
class ParkingLot(Base):
    __tablename__ = 'parking_lots'  # 映射到数据库中的表名

    parking_lot_id = Column(Integer, primary_key=True)  # 停车场唯一标识符，主键
    name = Column(String, nullable=False)  # 停车场名称，不能为空
    address = Column(String, nullable=False)  # 停车场地址，不能为空
    total_spaces = Column(Integer, default=0, nullable=False)  # 总车位数，默认值为0，不能为空
    available_spaces = Column(Integer, default=0, nullable=False)  # 可用车位数，默认值为0，不能为空
    hourly_rate = Column(Numeric(8, 2), nullable=False)  # 每小时收费标准，精确到分(8位总数，2位小数)

    # 定义与 parking_space 表的关系，'spaces' 是 ParkingLot 实例可以访问的属性
    spaces = relationship("ParkingSpace", back_populates="lots")

    def __repr__(self):
        return f"<ParkingLot(parking_lot_id='{self.parking_lot_id}', name='{self.name}', available_spaces='{self.available_spaces}')>"

# 定义车位表
class ParkingSpace(Base):
    __tablename__ = 'parking_spaces'

    space_id = Column(Integer, primary_key=True)  # 车位唯一标识符，主键
    space_number = Column(String, nullable=False)  # 车位编号(如"A区001")，不能为空
    space_type = Column(String, default='standard', nullable=False)  # 车位类型：标准、残疾人、VIP、电动车，默认为标准车位
    current_status = Column(String, default='available', nullable=False)  # 当前状态：可用、占用、预留、维护中，默认为可用
    vehicle_id = Column(String, default=None)  # 当前停放车辆ID，默认为空(无车辆)

    # 定义外键，关联到 authors 表的 author_id
    parking_lot_id = Column(Integer, ForeignKey('parking_lots.parking_lot_id', ondelete='CASCADE'), nullable=False)

    # 定义与 ParkingLot 表的关系，'lots' 是 ParkingSpace 实例可以访问的属性
    lots = relationship("ParkingLot", back_populates="spaces")
    # 定义与 ParkingRecord 表的关系，'records' 是 ParkingSpace 实例可以访问的属性
    records = relationship("ParkingRecord", back_populates="spaces")

    # 添加检查约束
    __table_args__ = (
        CheckConstraint(
            "space_type IN ('standard', 'disabled', 'VIP', 'electric')",
            name='check_space_type'
        ),
        CheckConstraint(
            "current_status IN ('available', 'occupied', 'reserved', 'maintenance')",
            name='check_current_status'
        ),
    )

    def __repr__(self):
        return f"<ParkingSpace(space_id={self.space_id}, space_number='{self.space_number}', status='{self.current_status}')>"

# 定义停车记录表
class ParkingRecord(Base):
    __tablename__ = 'parking_records'

    parking_record_id = Column(Integer, primary_key=True, autoincrement=True)  # 停车记录唯一标识，主键且自动递增
    vehicle_license = Column(String, nullable=False)  # 车牌号码，不能为空
    entry_time = Column(DateTime, nullable=False)  # 入场时间，不能为空
    exit_time = Column(DateTime, default=None)  # 出场时间，默认为空(表示车辆尚未出场)
    total_cost = Column(Numeric(8, 2), default=0.00, nullable=False)  # 总费用，默认为0.00
    payment_status = Column(String, default='available', nullable=False)   # 支付状态：未支付、已支付、已取消，默认为未支付
    created_at = Column(DateTime, server_default=func.now(), nullable=False)  # 记录创建时间，默认为当前时间戳
    parking_lot_id = Column(Integer, ForeignKey('parking_lots.parking_lot_id', ondelete='CASCADE'), nullable=False)  # 停车场ID，外键关联parking_lots表
    space_id = Column(Integer, ForeignKey('parking_spaces.space_id', ondelete='CASCADE'), nullable=False)  # 使用车位ID，外键关联parking_spaces表

    # 定义与 ParkingLot 表的关系，'lots' 是 ParkingRecord 实例可以访问的属性
    lots = relationship("ParkingLot")
    # 定义与 ParkingSpace 表的关系，'spaces' 是 ParkingRecord 实例可以访问的属性
    spaces = relationship("ParkingSpace", back_populates="records")

    # 添加数据库级别的检查约束
    __table_args__ = (
        # 确保 payment_status 字段只能是这几个值之一
        CheckConstraint(
            "payment_status IN ('unpaid', 'paid', 'cancelled')",
            name='check_payment_status'
        ),
    )

    def __repr__(self):
        return f"<ParkingRecord(record_id={self.parking_record_id}, vehicle='{self.vehicle_license}', status='{self.payment_status}')>"


# --- 创建数据库和表 ---
# 这一步会根据上面定义的模型，在数据库中创建相应的表
Base.metadata.create_all(engine)
print("数据库和表已成功创建。")

# 创建会话（Session）
# Session 是我们与数据库进行所有交互的接口
Session = sessionmaker(bind=engine)
session = Session()

# --- 示例一：插入数据 (Create) ---
print("\n--- 插入数据 ---")
# 实例化模型对象
parking_lot1 = ParkingLot(
    parking_lot_id=1,
    name="停车场1",
    address="位置1",
    total_spaces=200,
    available_spaces=150,
    hourly_rate=5.00
)
parking_lot2 = ParkingLot(
    parking_lot_id=2,
    name="停车场2",
    address="位置2",
    total_spaces=300,
    available_spaces=250,
    hourly_rate=6.00
)
parking_lot3 = ParkingLot(
    parking_lot_id=3,
    name="停车场3",
    address="位置1",
    total_spaces=100,
    available_spaces=80,
    hourly_rate=4.00
)

# 将对象添加到会话中
session.add_all([parking_lot1, parking_lot2, parking_lot3])

# 插入车位信息
parking_spaces = [
    ParkingSpace(space_id=101, parking_lot_id=1, space_number='A001', space_type='standard', current_status='available', vehicle_id=None),  # 标准车位，可用
    ParkingSpace(space_id=102, parking_lot_id=1, space_number='A002', space_type='standard', current_status='available', vehicle_id=None),  # 标准车位，可用
    ParkingSpace(space_id=103, parking_lot_id=1, space_number='A003', space_type='standard', current_status='occupied', vehicle_id='京A12345'),  # 标准车位，已被车牌京A12345占用
    ParkingSpace(space_id=104, parking_lot_id=1, space_number='B001', space_type='VIP', current_status='available', vehicle_id=None),  # VIP车位，可用
    ParkingSpace(space_id=105, parking_lot_id=1, space_number='C001', space_type='disabled', current_status='available', vehicle_id=None),  # 残疾人车位，可用
    ParkingSpace(space_id=106, parking_lot_id=1, space_number='D001', space_type='electric', current_status='maintenance', vehicle_id=None),  # 电动车位，维护中
    ParkingSpace(space_id=201, parking_lot_id=2, space_number='P001', space_type='standard', current_status='available', vehicle_id=None),  # 标准车位，可用
    ParkingSpace(space_id=202, parking_lot_id=2, space_number='P002', space_type='standard', current_status='available', vehicle_id=None),  # 标准车位，可用
    ParkingSpace(space_id=203, parking_lot_id=2, space_number='P003', space_type='VIP', current_status='reserved', vehicle_id=None),  # VIP车位，已预留
    ParkingSpace(space_id=204, parking_lot_id=2, space_number='P004', space_type='electric', current_status='available', vehicle_id=None),  # 电动车位，可用
    ParkingSpace(space_id=301, parking_lot_id=3, space_number='L001', space_type='standard', current_status='available', vehicle_id=None),  # 标准车位，可用
    ParkingSpace(space_id=302, parking_lot_id=3, space_number='L002', space_type='standard', current_status='occupied', vehicle_id='京B54321'),  # 标准车位，已被车牌京B54321占用
    ParkingSpace(space_id=303, parking_lot_id=3, space_number='L003', space_type='standard', current_status='available', vehicle_id=None)  # 标准车位，可用
]

session.add_all(parking_spaces)

# 创建停车记录实例
parking_records = [
    ParkingRecord(
        vehicle_license='京A12345',
        parking_lot_id=1,
        space_id=103,
        entry_time=datetime(2023, 10, 15, 9, 30, 0),
        exit_time=datetime(2023, 10, 15, 12, 45, 0),
        total_cost=16.50,
        payment_status='paid'
    ),
    ParkingRecord(
        vehicle_license='京C98765',
        parking_lot_id=2,
        space_id=201,
        entry_time=datetime(2023, 10, 15, 10, 15, 0),
        exit_time=datetime(2023, 10, 15, 11, 30, 0),
        total_cost=9.00,
        payment_status='paid'
    ),
    ParkingRecord(
        vehicle_license='京D55555',
        parking_lot_id=3,
        space_id=302,
        entry_time=datetime(2023, 10, 15, 8, 0, 0),
        exit_time=datetime(2023, 10, 15, 18, 0, 0),
        total_cost=40.00,
        payment_status='paid'
    ),
    ParkingRecord(
        vehicle_license='京E12345',
        parking_lot_id=1,
        space_id=101,
        entry_time=datetime(2023, 10, 15, 14, 20, 0),
        payment_status='unpaid'
    ),
    ParkingRecord(
        vehicle_license='京F67890',
        parking_lot_id=2,
        space_id=202,
        entry_time=datetime(2023, 10, 15, 13, 45, 0),
        payment_status='unpaid'
    ),
    ParkingRecord(
        vehicle_license='京G11111',
        parking_lot_id=1,
        space_id=102,
        entry_time=datetime(2023, 10, 14, 15, 0, 0),
        exit_time=datetime(2023, 10, 14, 15, 10, 0),
        payment_status='cancelled'
    )
        ]
session.add_all(parking_records)

# 提交所有更改到数据库
session.commit()
print("数据已成功插入。")

# --- 示例二：查询数据 (Read) ---
print("\n--- 查询所有停车场和它们的所停过车辆 ---")
# ORM 方式的 JOIN 查询
# 我们可以直接通过对象的属性来查询关联数据
results = session.query(ParkingRecord).join(ParkingLot).all()
for record in results:
    print(f"停车场名: {record.lots.name}, 车牌号: {record.vehicle_license}")

# --- 示例三：更新和删除数据 (Update & Delete) ---
print("\n--- 更新车辆出库 ---")
# 查询要更新的对象
record_to_update = session.query(ParkingRecord).filter_by(vehicle_license='京E12345').first()
if record_to_update:
    record_to_update.exit_time = datetime(2023, 10, 15, 18, 0, 0)
    record_to_update.total_cost = 25.00
    session.commit()
    print("更新'京E12345'出库。")

# 再次查询，验证更新
updated_parking = session.query(ParkingRecord).filter_by(vehicle_license='京E12345').first()
if updated_parking:
    print(f"更新后的信息: 车辆: {updated_parking.vehicle_license}, 出库时间: {updated_parking.exit_time}, 费用: {updated_parking.total_cost}")

print("\n--- 删除车辆信息 ---")
# 查询要删除的对象
record_to_delete = session.query(ParkingRecord).filter_by(vehicle_license='京G11111').first()
if record_to_delete:
    session.delete(record_to_delete)
    session.commit()
    print("车辆 '京G11111' 已被删除。")

# 再次查询借阅人列表，验证删除操作
print("\n--- 剩余的车辆信息 ---")
remaining_record = session.query(ParkingRecord).all()
for vehicle_license in remaining_record:
    print(f"车牌号: {vehicle_license.vehicle_license}")

# 关闭会话
session.close()
print("\n会话已关闭。")
