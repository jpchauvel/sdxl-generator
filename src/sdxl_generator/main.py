#!/usr/bin/env python
"""
Generate ten SDXL images with a local LoRA at 0.5 strength.
"""
import argparse
import os
import pathlib

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

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
        "--lora_paths",
        nargs="*",
        default=[],
        help="List of LoRA .safetensors files.",
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
        "--width", type=int, default=1024, help="Image width (default 1024)."
    )
    p.add_argument(
        "--lora_strengths",
        nargs="*",
        type=float,
        default=[],
        help="List of LoRA strengths (same length as --lora_paths).",
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
# helper: Get pipeline embeds for prompts bigger than the maxlength
# --------------------------------------------------------------------
def get_pipeline_embeds(
    pipeline, prompt, negative_prompt, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:
    """
    max_length = pipeline.tokenizer.model_max_length

    # simple way to determine length of tokens
    input_ids = pipeline.tokenizer(
        prompt, return_tensors="pt", truncation=False
    ).input_ids.to(device)
    negative_ids = pipeline.tokenizer(
        negative_prompt, return_tensors="pt", truncation=False
    ).input_ids.to(device)

    # create the tensor based on which prompt is longer
    if input_ids.shape[-1] >= negative_ids.shape[-1]:
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipeline.tokenizer(
            negative_prompt,
            truncation=False,
            padding="max_length",
            max_length=shape_max_length,
            return_tensors="pt",
        ).input_ids.to(device)

    else:
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipeline.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
            padding="max_length",
            max_length=shape_max_length,
        ).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(
            pipeline.text_encoder(input_ids[:, i : i + max_length])[0]
        )
        neg_embeds.append(
            pipeline.text_encoder(negative_ids[:, i : i + max_length])[0]
        )

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)


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
            use_safetensors=True,
        ).to(a.device)

    # 2 ▸ Load all specified LoRAs with matching strengths
    if len(a.lora_paths) != len(a.lora_strengths):
        raise ValueError(
            f"--lora_paths ({len(a.lora_paths)}) and --lora_strengths ({len(a.lora_strengths)}) must match in length."
        )

    if pipe.num_fused_loras > 0:
        print("Unfusing previously fused LoRA weights...")
        pipe.unfuse_lora()

    adapter_names = []
    for idx, (lora_path, strength) in enumerate(
        zip(a.lora_paths, a.lora_strengths)
    ):
        adapter_name = f"lora_{idx}"
        print(
            f"Loading LoRA {lora_path} with strength {strength} as adapter '{adapter_name}'"
        )
        pipe.load_lora_weights(
            lora_path, adapter_name=adapter_name, prefix=None
        )
        adapter_names.append(adapter_name)

    if adapter_names:
        pipe.set_adapters(adapter_names, adapter_weights=a.lora_strengths)

    print("Fusing LoRA into base weights...")
    pipe.fuse_lora()

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
        prompt_embeds, neg_prompt_embeds = get_pipeline_embeds(
            pipe,
            ", ".join([embed_prefix, system_prompt, sub]),
            neg_prompt,
            a.device,
        )

        g = torch.Generator(a.device).manual_seed(a.seed + i)

        images: list[Image.Image] = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_prompt_embeds,
            num_inference_steps=a.steps,
            guidance_scale=a.guidance_scale,
            height=a.width * ASPECT_RATIO[a.aspect_ratio],
            width=a.width,
            num_images_per_prompt=a.batch,
            generator=g,
        ).images

        for j, img in enumerate(images, 1):
            fname = os.path.join(a.out_dir, f"sdxl_{i:02d}_{j:02d}.png")
            img.save(fname)
            print(f"[{i}/10 – {j}/{a.batch}] → {fname}")

    print("Done!")


if __name__ == "__main__":
    main()
