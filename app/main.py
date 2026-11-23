
import json
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI, APIError
from pydantic import BaseModel, Field
from redis.asyncio import Redis


DATA_DIR = Path("/app/data")
DB_PATH = DATA_DIR / "agent.db"

# 以下全部是模拟数据，不代表真实设备状态
MOCK_DEVICES = {
    "wan0": {
        "status": "down",
        "optical_power_dbm": -28.7,
        "alarms": ["LOS"],
    },
    "lan0": {
        "status": "up",
        "optical_power_dbm": -12.5,
        "alarms": [],
    },
}

TOOL_NAMES = {
    "get_interface_status": "查询接口运行状态",
    "get_optical_power": "查询接口光功率，单位 dBm",
    "get_active_alarms": "查询当前接口告警",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "interface": {
                        "type": "string",
                        "enum": ["wan0", "lan0"],
                    }
                },
                "required": ["interface"],
                "additionalProperties": False,
            },
        },
    }
    for name, description in TOOL_NAMES.items()
]


def execute_tool(name: str, arguments: dict) -> dict:
    interface = arguments.get("interface")
    device = MOCK_DEVICES.get(interface)

    if device is None:
        return {"error": "Unknown interface"}

    if name == "get_interface_status":
        result = {"status": device["status"]}
    elif name == "get_optical_power":
        result = {
            "optical_power_dbm": device["optical_power_dbm"]
        }
    elif name == "get_active_alarms":
        result = {"alarms": device["alarms"]}
    else:
        return {"error": "Unknown tool"}

    return {
        "interface": interface,
        "source": "mock_fixture",
        **result,
    }


def get_history(session_id: str) -> list:
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        rows = conn.execute(
            """
            SELECT question, answer
            FROM history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT 4
            """,
            (session_id,),
        ).fetchall()

    messages = []

    for question, answer in reversed(rows):
        messages.append({
            "role": "user",
            "content": question,
        })
        messages.append({
            "role": "assistant",
            "content": answer,
        })

    return messages


def save_history(
    session_id: str,
    question: str,
    answer: str,
    tool_calls: list,
):
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute(
            """
            INSERT INTO history
            (session_id, question, answer, tools)
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                question,
                answer,
                json.dumps(tool_calls, ensure_ascii=False),
            ),
        )
        conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                tools TEXT NOT NULL
            )
        """)

    redis = Redis.from_url(
        os.environ["REDIS_URL"],
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )
    await redis.ping()

    key_path = Path(os.environ["DEEPSEEK_API_KEY_FILE"])
    api_key = key_path.read_text(encoding="utf-8").strip()

    if not api_key:
        raise RuntimeError("DeepSeek API key is empty")

    llm = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=45,
        max_retries=1,
    )

    app.state.redis = redis
    app.state.llm = llm

    try:
        yield
    finally:
        await llm.close()
        await redis.aclose()


app = FastAPI(
    title="Docker Day 10 Telecom Agent",
    version="1.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{32}$",
    )


@app.get("/health")
async def health(request: Request):
    try:
        await request.app.state.redis.ping()

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("SELECT 1")

        return {"status": "healthy"}

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Dependency unavailable",
        )


@app.get("/metrics")
async def metrics(request: Request):
    count = await request.app.state.redis.get(
        "day10:request_count"
    )
    return {"request_count": int(count or 0)}


@app.get("/sessions/{session_id}")
async def session_history(session_id: str):
    return {
        "session_id": session_id,
        "messages": get_history(session_id),
    }


@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    session_id = body.session_id or uuid4().hex

    messages = [
        {
            "role": "system",
            "content": (
                "你是通信网络故障诊断助手。"
                "当用户要求诊断接口时，先调用相关工具获取证据。"
                "所有工具数据均为模拟数据，必须明确说明。"
                "禁止编造设备状态，禁止执行真实设备操作。"
            ),
        }
    ]

    messages.extend(get_history(session_id))
    messages.append({
        "role": "user",
        "content": body.message,
    })

    used_tools = []
    answer = None

    try:
        for _ in range(4):
            response = await request.app.state.llm.chat.completions.create(
                model=os.getenv(
                    "DEEPSEEK_MODEL",
                    "deepseek-flash",
                ),
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                extra_body={
                    "thinking": {"type": "disabled"}
                },
            )

            result = response.choices[0].message

            if not result.tool_calls:
                answer = result.content
                break

            messages.append({
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": [
                    call.model_dump(exclude_none=True)
                    for call in result.tool_calls
                ],
            })

            for call in result.tool_calls:
                try:
                    arguments = json.loads(
                        call.function.arguments or "{}"
                    )
                    if not isinstance(arguments, dict):
                        arguments = {}
                except json.JSONDecodeError:
                    arguments = {}

                tool_result = execute_tool(
                    call.function.name,
                    arguments,
                )

                used_tools.append({
                    "name": call.function.name,
                    "arguments": arguments,
                    "result": tool_result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                    ),
                })

    except APIError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM error: {type(exc).__name__}",
        ) from exc

    if not answer:
        raise HTTPException(
            status_code=502,
            detail="Agent did not produce a final answer",
        )

    save_history(
        session_id,
        body.message,
        answer,
        used_tools,
    )

    count = await request.app.state.redis.incr(
        "day10:request_count"
    )

    return {
        "session_id": session_id,
        "answer": answer,
        "tool_calls": used_tools,
        "request_count": count,
        "data_source": "mock_fixture",
    }

@app.get("/version")
async def version():
    return {
        "version": os.getenv("APP_VERSION", "dev"),
        "release_note": "Added deployment metadata"
    }