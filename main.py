import argparse
import os
import math
from PIL import Image, ImageEnhance
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler
)

# イージング関数定義
def linear(t): return t
def ease_in(t): return t * t
def ease_out(t): return 1 - (1 - t) * (1 - t)
def ease_in_out(t): return -(math.cos(math.pi * t) - 1) / 2

EASING_FUNCTIONS = {
    "linear": linear,
    "ease-in": ease_in,
    "ease-out": ease_out,
    "ease-in-out": ease_in_out,
}

# 前処理：コントラスト・シャープネス
def preprocess_image(image, contrast_factor=1.0, sharpness_factor=1.0):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness_factor)

    return image

# 線画抽出＆動的thresholdで補間
def interpolate_lines(img1: Image.Image, img2: Image.Image, t: float) -> Image.Image:
    def extract_lines(image: Image.Image, threshold: int = 128) -> Image.Image:
        gray = image.convert("L")
        binary = gray.point(lambda p: 255 if p > threshold else 0)
        return binary

    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    # 時間によるしきい値変化
    thresholdA = int(lerp(100, 150, t))  # Aはだんだん薄く
    thresholdB = int(lerp(150, 100, t))  # Bはだんだん濃く

    lines1 = extract_lines(img1, thresholdA)
    lines2 = extract_lines(img2, thresholdB)

    # 混合（A寄り〜B寄りへ）
    blended = Image.blend(lines1, lines2, t)
    return blended.convert("RGB")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch.float32
    ).to(device)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # 画像読み込み＆前処理
    image1_path = os.path.join(args.datadir, "00000.png")
    image2_path = os.path.join(args.datadir, "00001.png")
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        raise FileNotFoundError("入力画像が見つかりません")

    raw_image_1 = Image.open(image1_path).convert("RGB")
    raw_image_2 = Image.open(image2_path).convert("RGB")

    image_1 = preprocess_image(raw_image_1, args.contrast, args.sharpness)
    image_2 = preprocess_image(raw_image_2, args.contrast, args.sharpness)

    os.makedirs(args.outputdir, exist_ok=True)
    easing_fn = EASING_FUNCTIONS.get(args.easing.lower(), linear)

    for i in range(args.num_frames):
        t = i / (args.num_frames - 1)
        eased_t = easing_fn(t)
        save_path = os.path.join(args.outputdir, f"{i:05d}.png")

        # 最初と最後はAI生成にかけず保存
        if i == 0:
            image_1.save(save_path)
            print(f"Saved original A: {save_path}")
            continue
        elif i == args.num_frames - 1:
            image_2.save(save_path)
            print(f"Saved original B: {save_path}")
            continue

        # 中間フレームは線画補間 → ControlNetで生成
        interpolated_line = interpolate_lines(image_1, image_2, eased_t)

        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=interpolated_line,
            controlnet_conditioning_scale=args.controlnet_strength,
            num_inference_steps=30
        ).images[0]

        result.save(save_path)
        print(f"Saved generated: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="./data/lineart", help="線画画像のディレクトリ")
    parser.add_argument("--outputdir", type=str, default="./outputs/lineart", help="補間結果の保存先")
    parser.add_argument("--num_frames", type=int, default=8, help="生成する補間フレーム数")
    parser.add_argument("--prompt", type=str, default="anime boy, character design, clean lineart, head and shoulders, facing sideways", help="生成プロンプト")
    parser.add_argument("--negative_prompt", type=str, default="girl, adult, messy lines, low quality", help="ネガティブプロンプト")
    parser.add_argument("--controlnet_strength", type=float, default=1.0, help="ControlNetの強さ（例：0.5〜1.5）")
    parser.add_argument("--easing", type=str, default="linear", choices=["linear", "ease-in", "ease-out", "ease-in-out"], help="補間イージングの種類")
    parser.add_argument("--contrast", type=float, default=1.0, help="コントラストの強さ（例：1.0〜2.0）")
    parser.add_argument("--sharpness", type=float, default=1.0, help="シャープネスの強さ（例：1.0〜2.0）")

    args = parser.parse_args()
    main(args)
