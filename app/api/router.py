"""
API 路由总装。

分文件而不是全塞在 main.py 里：路由按业务域拆分（认证/文档/问答/管理/健康），
每个文件职责单一，出问题能快速定位，也便于多人协作时减少冲突。
"""

from fastapi import APIRouter

from app.api.v1 import admin, auth, chat, documents, health

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(admin.router)
# 健康检查挂在根路径（/health、/ready），不放在 /api/v1 下：
# 探针地址应当稳定，不应该随 API 版本变化
health_router = health.router
