#个人觉得ORM框架更好用，就用的ORM
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

engine = create_engine('sqlite:///idol_orm.db', echo=True)#创sqlite数据库引擎
Base = declarative_base()#创ORM模型基类

class Idol(Base):
    __tablename__ = 'idols'
    idol_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    height = Column(Integer)
    song = relationship("Song", back_populates="idol")

class Song(Base):
    __tablename__ = 'songs'
    song_id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    idol_id = Column(Integer, ForeignKey('idols.idol_id'))
    idol = relationship("Idol", back_populates="song")    

Base.metadata.create_all(engine)
print("数据库和表已成功创建。")

# 创建会话（Session）
# Session 是我们与数据库进行所有交互的接口
Session = sessionmaker(bind=engine)
session = Session()

print("\n--- 插入数据 ---")
# 实例化模型对象
hokuto = Idol(name="冰鹰北斗", height=173)
subaru = Idol(name="明星昴流", height=172)
mao = Idol(name="衣更真绪", height=171)

# 将对象添加到会话中
session.add_all([hokuto, subaru, mao])

song_mao = Song(title="取景框中的你", idol=mao)
song_xing = Song(title="大爆炸之星", idol=subaru)
song_bei = Song(title="浪漫圣夜", idol=hokuto)
session.add_all([song_mao, song_xing, song_bei])

# 提交所有更改到数据库
session.commit()
print("数据已成功插入。")

#进行歌曲偶像查询
print("\n--- 查询数据 ---")
results = session.query(Song).join(Idol).all()
for song in results:
    print(f"歌曲：{song.title}，ByIdol：{song.idol.name}")

