import json
import time
import logging
from typing import Any, Dict, List, BinaryIO
from io import BytesIO

import requests

from .models import GenerationConfig, Message, GPTTaskResponse, TaskStatus

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
        status = resp_json["data"]["status"]

        return TaskStatus(status)


def get_task_result(url: str, client_id: str, task_id: int, dst: BinaryIO):
    with requests.get(
        f"{url}/v1/inference_tasks/{client_id}/{task_id}/images/0", stream=True
    ) as resp:
        resp.raise_for_status()

        for chunk in resp.iter_content(chunk_size=8192):
            dst.write(chunk)


def run_gpt_task(
    url: str,
    model: str,
    messages: List[Message],
    generation_config: GenerationConfig,
    seed: int,
    task_timeout: int = 600
) -> str:
    client_id = "abcd"

    # create task
    task_args = {
        "model": model,
        "messages": messages,
        "generation_config": generation_config,
        "seed": seed,
    }
    task_type = 1

    task_id = create_task(url, client_id, task_args, task_type)
    logging.info(f"create task {task_id} success")

    # check task status
    deadline = time.time() + task_timeout
    while time.time() < deadline:
        status = get_task_status(url, client_id, task_id)
        logging.debug(f"task {task_id} status: {status}")
        if status == TaskStatus.TaskSuccess:
            break
        time.sleep(1)

    logging.info(f"task {task_id} success")

    # download result file
    with BytesIO() as dst:
        get_task_result(url, client_id, task_id, dst)

        resp: GPTTaskResponse = json.loads(dst.getvalue())
    logging.info(f"get task {task_id} result success")

    assert len(resp["choices"]) > 0
    return resp["choices"][0]["message"]["content"]
