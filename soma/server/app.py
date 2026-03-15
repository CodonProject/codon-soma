import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


_logger = logging.getLogger(__name__)


app = FastAPI()

_cors_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:3000,http://localhost:8000').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'PUT', 'DELETE'],
    allow_headers=['Content-Type', 'Authorization'],
)

_connected_clients: List[WebSocket] = []

_MAX_MESSAGE_SIZE = 1024 * 1024


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket) -> None:
    '''
    WebSocket endpoint for real-time communication.

    Accepts connections and maintains a list of connected clients.
    Handles incoming messages and disconnections.

    Args:
        websocket (WebSocket): The WebSocket connection instance.
    '''
    await websocket.accept()
    _connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if len(data) > _MAX_MESSAGE_SIZE:
                continue
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            await _handle_client_message(websocket, message)
    except WebSocketDisconnect:
        if websocket in _connected_clients:
            _connected_clients.remove(websocket)


async def _handle_client_message(websocket: WebSocket, message: Dict[str, Any]) -> None:
    '''
    Handles incoming messages from WebSocket clients.

    Args:
        websocket (WebSocket): The WebSocket connection that sent the message.
        message (Dict[str, Any]): The parsed message from the client.
    '''
    pass


async def broadcast_state(state: Dict[str, Any]) -> None:
    '''
    Broadcasts state to all connected WebSocket clients.

    Sends the state dictionary as JSON to each connected client.
    Handles disconnections gracefully by removing disconnected clients.

    Args:
        state (Dict[str, Any]): The state dictionary to broadcast.
    '''
    if not _connected_clients:
        return

    message = json.dumps(state)
    disconnected = []

    for client in _connected_clients:
        try:
            await client.send_text(message)
        except (ConnectionError, RuntimeError) as e:
            _logger.warning('WebSocket client disconnected: %s', e)
            disconnected.append(client)

    for client in disconnected:
        if client in _connected_clients:
            _connected_clients.remove(client)


def get_static_dir() -> Path:
    '''
    Returns the path to the static files directory.

    Returns:
        Path: Path to the static directory relative to this file.
    '''
    return Path(__file__).parent / 'static'


_static_dir = get_static_dir()
if _static_dir.exists():
    app.mount('/static', StaticFiles(directory=str(_static_dir)), name='static')
