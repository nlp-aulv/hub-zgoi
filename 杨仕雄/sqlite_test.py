import sqlite3

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# 连接到数据库，如果文件不存在会自动创建
conn = sqlite3.connect('students.db')
cursor = conn.cursor()

# 创建 student 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    name TEXT PRIMARY KEY,
    age INTEGER NOT NULL,
    sex TEXT NOT NULL
);
''')

# 提交更改
conn.commit()
print("数据库和表已成功创建。")

# 增
cursor.execute("INSERT INTO students(name,age,sex) VALUES (?,?,?)",('TOM',20,'男'))
cursor.execute("INSERT INTO students(name,age,sex) VALUES (?,?,?)",('AMY',22,'女'))

conn.commit()

print('查询所有学生')
cursor.execute('''
SELECT *
FROM students;
''')

result = cursor.fetchall()
print(result)

# 删
cursor.execute("DELETE FROM  students WHERE name = ?",('AMY',))

print('delete----查询所有学生')
cursor.execute('''
SELECT *
FROM students;
''')
result2 = cursor.fetchall()
print(result2)

# 改
cursor.execute("UPDATE  students set age = 30 WHERE name = ?",('TOM',))

print('update----查询所有学生')
cursor.execute('''
SELECT *
FROM students;
''')
result3 = cursor.fetchall()
print(result3)

# 清空表
print('清空表')
cursor.execute("DELETE FROM  students")
conn.commit()

print('delete----查询所有学生')
cursor.execute('''
SELECT *
FROM students;
''')
result4 = cursor.fetchall()
print(result4)

# 关闭连接
conn.close()
print("\n数据库连接已关闭。")


print('-------ORM--------')
# 创建数据库引擎，这里使用 SQLite
# check_same_thread=False 允许在多线程环境下使用，但对于单文件示例可以忽略
engine = create_engine('sqlite:///student_orm.db', echo=True)

# 创建 ORM 模型的基类
Base = declarative_base()

class Student(Base):
    __tablename__ = 'student'

    name = Column(String, primary_key=True)
    age = Column(Integer, nullable=False)
    sex = Column(String, nullable=False)

    def __repr__(self):
        return f"<Student(name='{self.name}', age='{self.age}' , sex ='{self.sex}')>"


# --- 创建数据库和表 ---
# 这一步会根据上面定义的模型，在数据库中创建相应的表
Base.metadata.create_all(engine)
print("数据库和表已成功创建。")

# 创建会话（Session）
# Session 是我们与数据库进行所有交互的接口
Session = sessionmaker(bind=engine)
session = Session()

# 清空表
session.query(Student).delete()
session.commit()

# --- 示例一：插入数据 (Create) ---
print("\n--- 插入数据 ---")
# 实例化模型对象
TOM = Student(name='TOM', age=20 , sex='男')
JACK = Student(name='JACK', age=20 , sex='男')
AMY = Student(name='AMY', age=22 , sex='女')
# 将对象添加到会话中
session.add_all([TOM,JACK,AMY])

# 提交所有更改到数据库
session.commit()
print("数据已成功插入。")

print("\n--- select ---")
results = session.query(Student).all()
print(results)

print("\n--- 更新 ---")
update_age = session.query(Student).filter_by(name='JACK').first()
if update_age:
    update_age.age = 30
    session.commit()
    print('JACK年龄已更新')

update_stu = session.query(Student).filter_by(name='JACK').first()
if update_stu:
    print(f'JACK更新后：{update_stu.age}')

print("\n--- 删除 ---")
# 查询要删除的对象
stu_to_delete = session.query(Student).filter_by(name='JACK').first()
if stu_to_delete:
    session.delete(stu_to_delete)
    session.commit()
    print("'JACK' 已被删除。")


# 再次查询借阅人列表，验证删除操作
print("\n--- 剩余stu ---")
stus = session.query(Student).all()
for stu in stus:
    print(f"姓名: {stu.name}")


# 关闭会话
session.close()
print("\n会话已关闭。")




