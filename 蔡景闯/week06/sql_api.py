from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

# 连接sqlite
engine = create_engine('sqlite:///UserSys.db', echo=False) # echo=True表示输出sql语句
# 创建对象的基类:
Base = declarative_base()


class User(Base):
    # 表的名字:
    __tablename__ = 'user'

    # 表的结构:
    id = Column(String(20), primary_key=True)
    username = Column(String(20))
    password = Column(String(20))

    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

    def __repr__(self):
        return f"<User(id='{self.id}', username='{self.username}', password='{self.password}')>"

class Friend(Base):
    # 表的名字:
    __tablename__ = 'friend'

    # 表的结构:
    id_user1 = Column(String(20), primary_key=True)
    id_user2 = Column(String(20), primary_key=True)

    def __init__(self, id_user1, id_user2):
        self.id_user1 = id_user1
        self.id_user2 = id_user2

    def __repr__(self):
        return f"<Friend(id_user1='{self.id_user1}', id_user2='{self.id_user2}')>"

# 创建表
Base.metadata.create_all(engine)

# 插入用户
def insert_user(user: User):

    """
    向数据库中插入新用户信息

    参数:
        user: User类型对象，包含用户id、用户名和密码等信息

    返回:
        无返回值
    """
    # 创建DBSession类型: 用于创建与数据库的会话
    DBSession = sessionmaker(bind=engine)
    # 创建session对象: 用于操作数据库的会话实例
    session = DBSession()
    # 创建新User对象: 使用传入的用户信息创建新的用户对象
    new_user = User(id=user.id, username=user.username, password=user.password)
    # 添加到session: 将新用户对象添加到会话中，准备写入数据库
    session.add(new_user)
    # 提交即保存到数据库: 提交会话，将更改写入数据库
    session.commit()
    # 关闭session: 关闭会话，释放资源
    session.close()

def insert_friend(user1: User, user2: User):
    """
    向数据库中添加好友关系
    参数:
        user1: 第一个用户对象
        user2: 第二个用户对象
    返回:
        无返回值
    功能:
        创建一个好友关系记录并保存到数据库
    """
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)
    # 创建session对象:
    session = DBSession()
    # 创建新User对象:
    new_friend = Friend(id_user1=user1.id, id_user2=user2.id)
    # 添加到session:
    session.add(new_friend)
    # 提交即保存到数据库:
    session.commit()
    # 关闭session:
    session.close()

def select_user(id: str):
    """
    根据用户ID查询用户信息
    参数:
        id (str): 用户ID
    返回:
        User: 返回查询到的用户对象，如果未找到则返回None
    """
    # 创建DBSession类型:
    # 这行代码创建了一个与数据库绑定的session工厂类
    DBSession = sessionmaker(bind=engine)
    # 创建session对象:
    # 这行代码创建了一个session实例，用于数据库操作
    session = DBSession()
    # 创建新User对象:
    # 这行代码使用session查询数据库，根据用户ID查找用户，并返回第一个匹配的结果
    user = session.query(User).filter(User.id == id).first()
    # 添加到session:
    # 这行代码被注释掉了，用于将新用户添加到session中
    # session.add(new_user)
    # 提交即保存到数据库:
    # 这行代码被注释掉了，用于提交session中的更改到数据库
    # session.commit()
    # 关闭session:
    # 这行代码关闭session，释放资源
    session.close()
    # 返回查询到的用户对象
    return user

def select_friends(id_user1: str):
    """
    根据用户ID查询该用户的所有好友信息
    参数:
        id_user1 (str): 要查询的用户ID
    返回:
        list: 返回该用户的所有好友记录列表
    """
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)
    # 创建session对象:
    session = DBSession()
    # 创建新User对象:
    friends = session.query(Friend).filter(Friend.id_user1 == id_user1).all()
    # 添加到session:
    # session.add(new_user)
    # 提交即保存到数据库:
    # session.commit()
    # 关闭session:
    session.close()
    return friends

def select_all_users():
    """
    查询所有用户信息
    返回:
        list: 返回所有用户记录列表
    """
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)
    # 创建session对象:
    session = DBSession()
    # 创建新User对象:
    users = session.query(User).all()
    # 添加到session:
    # session.add(new_user)
    # 提交即保存到数据库:
    # session.commit()
    # 关闭session:
    session.close()
    return users

def select_all_friends():
    """
    查询所有好友信息
    返回:
        list: 返回所有好友记录列表
    """
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)
    # 创建session对象:
    session = DBSession()
    # 创建新User对象:
    friends = session.query(Friend).all()
    # 添加到session:
    # session.add(new_user)
    # 提交即保存到数据库:
    # session.commit()
    # 关闭session:
    session.close()
    return friends


user1 = User("1", "user1", "password1")
user2 = User("2", "user2", "password2")
user3 = User("3", "user3", "password3")

# 插入数据
# print('插入数据')
# insert_user(user1)
# insert_user(user2)
# insert_user(user3)
# insert_friend(user1, user2)
# insert_friend(user1, user3)

# 查询数据
print('查询数据')
all_users = select_all_users()
user = select_user("1")
print(all_users)
print(user)
friend = select_friends("1")
print(friend)








