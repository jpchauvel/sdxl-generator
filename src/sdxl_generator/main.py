#!/usr/bin/env python
"""
Generate ten SDXL images with a local LoRA at 0.5 strength.
"""
import argparse
import os
import pathlib

import torch
from diffusers import StableDiffusionXLPipeline

ASPECT_RATIO = {
    "1:1": 1.0,
    "4:3": 3 / 4,
    "3:4": 4 / 3,
    "16:9": 9 / 16,
    "9:16": 16 / 9,
}


# --------------------------------------------------------------------
#  Argument parsing
# --------------------------------------------------------------------
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDXL text-to-image with one LoRA and multiple embeddings."
    )
    p.add_argument(
        "--ckpt_path",
        required=True,
        help="Folder / HF repo or single .safetensors checkpoint.",
    )
    p.add_argument(
        "--lora_path", required=True, help="LoRA folder or .safetensors file."
    )
    p.add_argument(
        "--out_dir", required=True, help="Where PNGs will be saved."
    )
    p.add_argument(
        "--system_prompt_path",
        required=True,
        help="Path to system prompt text file.",
    )
    p.add_argument(
        "--neg_prompt_path",
        required=True,
        help="Path to negative prompt text file.",
    )
    p.add_argument(
        "--sub_prompts_path",
        required=True,
        help="Path to sub-prompts text file. Prompts are separeted by newlines.",
    )
    p.add_argument(
        "--emb_paths",
        nargs="+",
        default=[],
        help="One or more textual-inversion embedding files.",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Images *per* prompt (num_images_per_prompt).",
    )
    p.add_argument(
        "--steps", type=int, default=30, help="Denoising steps (default 30)."
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed (index is added per prompt).",
    )
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale (default 7.5).",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=77,
        help="Max tokens per prompt (default 77).",
    )
    p.add_argument(
        "--width", type=int, default=1024, help="Image width (default 1024)."
    )
    p.add_argument(
        "--lora_strength",
        type=float,
        default=0.5,
        help="LoRA strength (default 0.5).",
    )
    p.add_argument(
        "--aspect_ratio",
        type=str,
        default="1:1",
        choices=ASPECT_RATIO.keys(),
        help="Image aspect ratio (default 1:1).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cpu", "mps", "cuda"],
        help="Device (default mps).",
    )
    return p.parse_args()


# --------------------------------------------------------------------
# helper: ensure prompt ≤ max_tokens
# --------------------------------------------------------------------
def trim_prompt(text: str, tokenizer, max_tokens: int = 75) -> str:
    """
    Clip a prompt so that it encodes to ≤ `max_tokens` (default 75) tokens.
    Keeps whole comma-separated fragments; lops them off from the right.
    """
    ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]
    if len(ids) <= max_tokens:
        return text  # already short enough

    # progressively drop the last comma-separated chunk
    new_text = ""
    parts = [p.strip() for p in text.split(",")]
    while parts and len(ids) > max_tokens:
        parts.pop()  # drop right-most chunk
        new_text = ", ".join(parts)
        ids = tokenizer(
            new_text, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
    return new_text


# --------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------
def main() -> None:
    a = cli()
    os.makedirs(a.out_dir, exist_ok=True)

    # 1 ▸ Build the SDXL pipeline
    is_file = os.path.isfile(a.ckpt_path) and a.ckpt_path.lower().endswith(
        ".safetensors"
    )
    if is_file:
        print(f"Loading single-file checkpoint: {a.ckpt_path}")
        # allows .safetensors [oai_citation:0‡huggingface.co](https://huggingface.co/docs/diffusers/v0.28.0/en/api/loaders/single_file?utm_source=chatgpt.com)
        pipe = StableDiffusionXLPipeline.from_single_file(
            a.ckpt_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(a.device)
    else:
        print(f"Loading directory / repo checkpoint: {a.ckpt_path}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            a.ckpt_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(a.device)

    # 2 ▸ Load the LoRA at 0.5
    pipe.load_lora_weights(a.lora_path, adapter_name="default_0", prefix=None)
    pipe.set_adapters(["default_0"], adapter_weights=[a.lora_strength])

    # 3 ▸ Load embeddings & collect tokens
    def load_embedding(path: str) -> str | None:
        """
        Try the standard diffusers TI loader first.
        If it fails with the typical 'clip_l / clip_g' SDXL file,
        fall back to loading both vectors and returning *one* token
        that is registered for **both** encoders.
        """
        try:
            # standard route
            return pipe.load_textual_inversion(path)
        except ValueError as e:
            # Fallback for Automatic1111-style SDXL embeddings
            token_name = f"<{pathlib.Path(path).stem}>"
            loaded = False
            for key in ("clip_l", "clip_g"):
                try:
                    pipe.load_textual_inversion(
                        path,
                        token=token_name,  # same token for both encoders
                        weight_name=key,  # pick the right vector
                    )
                    loaded = True
                except Exception:
                    pass
            if loaded:
                print(
                    f"Loaded SDXL dual-vector TI '{path}' as token {token_name}"
                )
                return token_name
            else:
                # Still unusable – warn and skip
                print(f"[skip] {path}: {e}")
                return None

    embed_tokens: list[str] = []
    for p in a.emb_paths:
        tok = load_embedding(p)
        if tok:
            embed_tokens.append(tok)

    embed_prefix = " ".join(embed_tokens)

    # 3.1 ▸ Load prompts
    with open(pathlib.Path(a.system_prompt_path).expanduser(), "r") as f:
        system_prompt = f.read()
    with open(pathlib.Path(a.neg_prompt_path).expanduser(), "r") as f:
        neg_prompt = f.read()
    with open(pathlib.Path(a.sub_prompts_path).expanduser(), "r") as f:
        sub_prompts = f.read().splitlines()

    # 4 ▸ Generate images
    for i, sub in enumerate(sub_prompts, 1):
        prompt = trim_prompt(
            ", ".join([embed_prefix, system_prompt, sub]),
            pipe.tokenizer,
            a.max_tokens,
        )
        neg_prompt = trim_prompt(neg_prompt, pipe.tokenizer, a.max_tokens)
        g = torch.Generator("mps").manual_seed(a.seed + i)

        images = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=a.steps,
            guidance_scale=a.guidance_scale,
            height=a.width * ASPECT_RATIO[a.aspect_ratio],
            width=a.width,
            num_images_per_prompt=a.batch,
            cross_attention_kwargs={"scale": a.lora_strength},
            generator=g,
        ).images

        for j, img in enumerate(images, 1):
            fname = os.path.join(a.out_dir, f"sdxl_{i:02d}_{j:02d}.png")
            img.save(fname)
            print(f"[{i}/10 – {j}/{a.batch}] → {fname}")

    print("Done!")


if __name__ == "__main__":
    main()
