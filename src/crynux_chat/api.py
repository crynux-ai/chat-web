import json
import logging
import time
import uuid
from io import BytesIO
from typing import Any, BinaryIO, Dict, List

import requests

from .error import TaskError
from .models import GenerationConfig, GPTTaskResponse, Message, TaskStatus

__all__ = ["run_gpt_task"]

_logger = logging.getLogger(__name__)


def create_task(
    url: str, client_id: str, task_args: Dict[str, Any], task_type: int
) -> int:
    task_args_str = json.dumps(task_args, ensure_ascii=False)
    with requests.post(
        f"{url}/v1/inference_tasks",
        json={
            "client_id": client_id,
            "task_args": task_args_str,
            "task_type": task_type,
        },
    ) as resp:
        resp.raise_for_status()

        resp_json = resp.json()
        task_id = resp_json["data"]["id"]

        return task_id


def get_task_status(url: str, client_id: str, task_id: int) -> TaskStatus:
    with requests.get(f"{url}/v1/inference_tasks/{client_id}/{task_id}") as resp:
        resp.raise_for_status()

        resp_json = resp.json()
        status = TaskStatus(resp_json["data"]["status"])

        if status == TaskStatus.TaskAborted:
            reason = resp_json["data"]["abort_reason"]
            _logger.error(f"Task {task_id} aborted for {reason}")

        return status


def get_task_result(url: str, client_id: str, task_id: int) -> GPTTaskResponse:
    with requests.get(
        f"{url}/v1/inference_tasks/{client_id}/gpt/{task_id}/result"
    ) as resp:
        resp.raise_for_status()

        resp_json = resp.json()
        data: GPTTaskResponse = resp_json["data"]
        return data


def run_gpt_task(
    url: str,
    model: str,
    messages: List[Message],
    generation_config: GenerationConfig,
    seed: int,
    task_timeout: int = 600,
) -> str:
    client_id = str(uuid.uuid4())

    # create task
    task_args = {
        "model": model,
        "messages": messages,
        "generation_config": generation_config,
        "seed": seed,
    }
    task_type = 1

    task_id = create_task(url, client_id, task_args, task_type)
    _logger.info(f"create task {task_id} success")

    # check task status
    deadline = time.time() + task_timeout
    status = TaskStatus.TaskPending
    while time.time() < deadline:
        status = get_task_status(url, client_id, task_id)
        _logger.debug(f"task {task_id} status: {status}")
        if status == TaskStatus.TaskSuccess:
            break
        elif status == TaskStatus.TaskAborted:
            raise TaskError("Task aborted")
        time.sleep(1)

    if status != TaskStatus.TaskSuccess:
        _logger.error(f"task {task_id} timeout")
        raise TaskError(f"Task timeout")

    _logger.info(f"task {task_id} success")

    # download result file
    resp = get_task_result(url, client_id, task_id)

    _logger.info(f"get task {task_id} result success")

    assert len(resp["choices"]) > 0
    return resp["choices"][0]["message"]["content"]
