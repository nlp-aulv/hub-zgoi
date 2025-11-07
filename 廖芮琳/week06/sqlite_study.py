# ai_prompt_manager.py
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime

# 数据库连接（SQLite 文件）
DATABASE_URL = "sqlite:///ai_prompt.db"
engine = create_engine(DATABASE_URL, echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# 模型定义

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True)
    templates = relationship("PromptTemplate", back_populates="creator")

    def __repr__(self):
        return f"<User(username={self.username})>"

class PromptTemplate(Base):
    __tablename__ = "prompt_templates"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))

    creator = relationship("User", back_populates="templates")
    records = relationship("PromptRecord", back_populates="template")

    def __repr__(self):
        return f"<PromptTemplate(title={self.title}, user={self.user_id})>"

class PromptRecord(Base):
    __tablename__ = "prompt_records"
    id = Column(Integer, primary_key=True)
    prompt_text = Column(Text)
    response_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String)
    template_id = Column(Integer, ForeignKey("prompt_templates.id"))

    template = relationship("PromptTemplate", back_populates="records")

    def __repr__(self):
        return f"<PromptRecord(model={self.model_name}, created={self.created_at})>"

# 创建表
Base.metadata.create_all(bind=engine)

# 创建 Session 并插入数据
session = SessionLocal()

# 添加用户
user = User(username="alice", email="alice@example.com")
session.add(user)
session.commit()

# 添加 Prompt 模板
template = PromptTemplate(
    title="Blog Outline Generator",
    content="Generate a blog outline about {topic}",
    creator=user
)
session.add(template)
session.commit()

# 添加一次调用记录
record = PromptRecord(
    prompt_text="Generate a blog outline about climate change",
    response_text="1. Introduction\n2. Causes\n3. Impacts\n4. Solutions",
    model_name="gpt-4",
    template=template
)
session.add(record)
session.commit()

# 查询与展示
templates = session.query(PromptTemplate).all()
for t in templates:
    print(f"\nPrompt: {t.title} by {t.creator.username}")
    for r in t.records:
        print(f" - {r.model_name} | {r.created_at:%Y-%m-%d %H:%M} | {r.prompt_text[:30]}...")

session.close()
