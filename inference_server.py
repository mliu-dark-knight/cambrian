import argparse
import os
import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
from fastapi import FastAPI, Request, BackgroundTasks

from cambrian.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, Any
from io import BytesIO
import uvicorn
import base64
from ezcolorlog import root_logger as logger

from PIL import Image
import math

# cambrian-phi3-3b
# conv_mode = "phi3"

# cambrian-8b
conv_mode = "llama_3"

# cambrian-34b
# conv_mode = "chatml_direct"

# cambrian-13b
# conv_mode = "vicuna_v1"


def process(image, question, tokenizer, image_processor, model_config):
    qs = question

    if model_config.mm_use_im_start_end:
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + qs
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_size = [image.size]
    image_tensor = process_images([image], image_processor, model_config)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    return input_ids, image_tensor, image_size, prompt


import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ModelWorker:
    def __init__(self, model_path: str, load_8bit=False, load_4bit=False) -> None:
        self.model_path = os.path.expanduser(model_path)
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path,
                None,
                self.model_name,
                load_8bit=load_8bit,
                load_4bit=load_4bit,
            )
        )
        self.temperature = 0

    def generate(self, image_str: str, prompt: str):
        image = Image.open(BytesIO(base64.b64decode(image_str)))
        input_ids, image_tensor, image_sizes, _ = process(
            image, prompt, self.tokenizer, self.image_processor, self.model.config
        )
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        yield outputs


app = FastAPI()


@app.post("/generate")
async def generate(request: Request):
    params = await request.json()
    response = worker.generate(params["image"], params["prompt"])
    return {"text": response}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=40000)
    parser.add_argument("--model-path", type=str, default="nyu-visionx/cambrian-8b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(
        args.model_path,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
