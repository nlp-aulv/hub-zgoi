from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# 创建数据库引擎，这里使用 SQLite
# check_same_thread=False 允许在多线程环境下使用，但对于单文件示例可以忽略
engine = create_engine('sqlite:///library_orm.db', echo=True)

# 创建 ORM 模型的基类
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'  # 映射到数据库中的表名

    user_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    password = Column(String, nullable=False)

    # 定义外键，关联到 authors 表的 author_id
    occ_id = Column(Integer, ForeignKey('occupations.occ_id'))

    occupations = relationship("Occupation", back_populates="users")

    def __repr__(self):
        return f"<Author(name='{self.name}', password='{self.password}')>"

#职位
class Occupation(Base):
    __tablename__ = 'occupations'

    occ_id = Column(Integer, primary_key=True)
    dept = Column(String, nullable=False)
    job = Column(String)

    users = relationship("User", back_populates="occupations")

    def __repr__(self):
        return f"<Book(dept='{self.dept}', job={self.job})>"

Base.metadata.create_all(engine)
print("数据库和表已成功创建。")

# 创建会话（Session）
# Session 是我们与数据库进行所有交互的接口
Session = sessionmaker(bind=engine)
session = Session()


print("\n--- 插入数据 ---")


normal_position = Occupation(dept='Safty', job="保安")
manager = Occupation(dept='Sales', job="销售经理")
session.add_all([normal_position, manager])

# 实例化模型对象
Anna = User(name='Anna', password='123456', occupations=normal_position)
Lisi = User(name='Lisihaha', password='lisi123', occupations=manager)
Bob = User(name='Bob123', password='123456', occupations=normal_position)

# 将对象添加到会话中
session.add_all([Anna, Lisi, Bob])

# 提交所有更改到数据库
session.commit()
print("数据已成功插入。")

results = session.query(User).join(Occupation).all()
for user in results:
    print(f"用户: {user.name}, 部门: {user.occupations.dept}, 职业: {user.occupations.job}")


# 关闭会话
session.close()
print("\n会话已关闭。")
