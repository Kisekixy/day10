# Docker Day 10 - Telecom Agent

基于 FastAPI、DeepSeek、Redis 和 SQLite 的
通信网络故障诊断 Agent 教学项目。

## Architecture

- FastAPI: Agent API
- DeepSeek: LLM
- Redis: Request Counter
- SQLite: Session History
- Docker Compose: Service Orchestration

## Requirements

- Docker Desktop
- Docker Compose
- DeepSeek API Key

## Configuration

创建：

secrets/deepseek_api_key.txt

写入 DeepSeek API Key。

## Start

docker compose up -d --build

## Stop

docker compose down

## API

GET /health

GET /metrics

POST /chat

GET /sessions/{session_id}

## Security

- API Key 不提交到 Git
- API Key 不写入 Dockerfile
- FastAPI 使用非 root 用户运行
- Redis 不暴露宿主机端口
- SQLite 使用 Docker Volume 持久化