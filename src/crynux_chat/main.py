from typing import List, Tuple

import gradio as gr

from .api import run_gpt_task
from .models import GenerationConfig, Message
from .config import get_config
from .log import init as log_init


class Server(object):
    def __init__(self, port: int, bridge_url: str, models: List[str]) -> None:
        self.port = port
        self.bridge_url = bridge_url
        self.models = models

    def chat(
        self,
        message: str,
        history: List[Tuple[str, str]],
        model: str,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: float,
    ):
        messages: List[Message] = []
        for user_content, bot_content in history:
            messages.extend(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": bot_content},
                ]
            )

        messages.append({"role": "user", "content": message})

        generation_config: GenerationConfig = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        res = run_gpt_task(
            url=self.bridge_url,
            model=model,
            messages=messages,
            generation_config=generation_config,
            seed=seed,
        )
        return res

    def launch(self):
        with gr.Blocks(title="Crynux") as demo:
            gr.HTML("""<h1 align="center">Crynux chat</h1>""")

            with gr.Row():
                with gr.Column(scale=1):
                    model_dropbox = gr.Dropdown(
                        choices=self.models, value=self.models[0], label="model"  # type: ignore
                    )
                    seed_slider = gr.Number(value=0, label="seed")

                    max_new_tokens_slider = gr.Slider(
                        0,
                        4096,
                        value=2048,
                        step=1,
                        label="max new tokens",
                        interactive=True,
                    )
                    temperature_slider = gr.Slider(
                        0, 2, value=1, step=0.01, label="temperature", interactive=True
                    )
                    top_p_slider = gr.Slider(
                        0, 1, value=1, step=0.01, label="top p", interactive=True
                    )
                    top_k_slider = gr.Slider(
                        0, 100, value=50, step=1, label="top k", interactive=True
                    )

                with gr.Column(scale=4):
                    gr.ChatInterface(
                        fn=self.chat,
                        additional_inputs=[
                            model_dropbox,
                            seed_slider,
                            max_new_tokens_slider,
                            temperature_slider,
                            top_p_slider,
                            top_k_slider,
                        ],
                        additional_inputs_accordion_name="generation config",
                    )

        demo.launch(share=False, inbrowser=True, server_port=self.port)


def main():
    config = get_config()
    log_init(config=config)
    server = Server(
        port=config.port, bridge_url=config.bridge_url, models=config.models
    )
    server.launch()


if __name__ == "__main__":
    main()
