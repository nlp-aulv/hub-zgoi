from elasticsearch.helpers import bulk

from user import UserCreate, UserUpdate, UserProfileCreate, UserProfileUpdate
from service import init_db, create_user, get_user, update_user, create_user_profile, get_user_profile
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

def main_db():
    # 初始化数据库
    init_db()

    # 创建新用户
    user = UserCreate(
        username="chen",
        password="chenyuhao",
        email="cyhasada@example.com"
    )
    user_id = create_user(user)
    print(f"✅ Created user with ID: {user_id}")

    # 创建用户详情
    profile = UserProfileCreate(
        user_id=user_id,
        real_name="John Doe",
        gender=1,
        age=30
    )
    create_user_profile(user_id, profile)
    print("✅ Created profile")

    # 获取用户详情
    user = get_user(user_id)
    print("\n🔍 User details:")
    print(f"ID: {user.id}, Username: {user.username}, Email: {user.email}")

    # 获取用户详情
    profile = get_user_profile(user_id)
    print("\n🔍 Profile details:")
    print(f"Real Name: {profile.real_name}, Age: {profile.age}, Gender: {profile.gender}")

    # 更新用户信息
    update_data = UserUpdate(
        email="chenyuhaoasd@example.com",
        status=0  # 禁用账户
    )
    update_user(user_id, update_data)
    print("\n🔄 Updated user status")

    # 验证更新
    updated_user = get_user(user_id)
    print(f"\n✅ Updated user: {updated_user.email}, Status: {updated_user.status}")


def main_es():

    # 连接Elasticsearch
    es = Elasticsearch(["http://localhost:9200"])

    # 创建索引并指定映射
    mapping = {
        "mappings": {
            "properties": {
                "name": {"type": "text"},
                "age": {"type": "integer"},
                "city": {"type": "keyword"},
                "job": {"type": "keyword"}
            }
        }
    }
    es.indices.create(index="users", body=mapping, ignore=400)

    # 插入数据
    users = [
        {"name": "Alice", "age": 25, "city": "New York", "job": "developer"},
        {"name": "Bob", "age": 32, "city": "San Francisco", "job": "designer"},
        {"name": "Charlie", "age": 45, "city": "Chicago", "job": "manager"}
    ]

    documents = [
        {"_index": "users", "_id": i + 1, "_source": user}
        for i, user in enumerate(users)
    ]

    bulk(es, documents)

    # 验证数据是否插入成功
    res = es.search(index="users", query={"match_all": {}})
    print(f"Total documents in index: {res['hits']['total']['value']}")

    # 搜索在纽约工作的人
    s = es.search(index="users", query={
        "match": {
            "city": "New York"
        }
    })
    print(f"Found {s['hits']['total']['value']} users in New York")

    # 搜索年龄大于30的设计师
    s = es.search(index="users", query={
        "bool": {
            "must": [
                {"range": {"age": {"gt": 30}}},
                {"term": {"job": "designer"}}
            ]
        }
    })
    print(f"Found {s['hits']['total']['value']} designers older than 30")

if __name__ == "__main__":
    #main_db()
    main_es()