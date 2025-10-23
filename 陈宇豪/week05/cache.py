from collections import OrderedDict
from threading import Lock
from typing import Any, Optional


class LRUMemoryKVStore:
    """LRU淘汰策略的内存键值存储"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._store = OrderedDict()
        self._lock = Lock()

    def set(self, key: str, value: Any) -> None:
        """设置键值对"""
        with self._lock:
            if key in self._store:
                # 更新现有键，移到末尾
                del self._store[key]
            elif len(self._store) >= self.max_size:
                # 删除最久未使用的项
                self._store.popitem(last=False)

            self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """获取值，并更新访问顺序"""
        with self._lock:
            if key not in self._store:
                return default

            # 移动到末尾（最近访问）
            value = self._store.pop(key)
            self._store[key] = value
            return value

    def delete(self, key: str) -> bool:
        """删除键值对"""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def peek(self, key: str, default: Any = None) -> Any:
        """查看值但不更新访问顺序"""
        with self._lock:
            return self._store.get(key, default)

    def keys(self) -> list:
        """获取所有键（按访问顺序）"""
        with self._lock:
            return list(self._store.keys())

    def clear(self) -> None:
        """清空存储"""
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        """获取当前大小"""
        with self._lock:
            return len(self._store)

    def is_full(self) -> bool:
        """检查是否已满"""
        with self._lock:
            return len(self._store) >= self.max_size