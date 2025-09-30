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


class TimeReversalInterpolator:
    """
    線画アニメ補間専用パイプライン
    - 最初と最後のフレームはオリジナルの線画を保存
    - 中間フレームは動的thresholdを用いた線画補間を経て ControlNet で生成
    """

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", controlnet_id="lllyasviel/sd-controlnet-scribble",
                 torch_dtype=torch.float32, device=None, controlnet_strength=1.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ControlNet とパイプラインの準備
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=torch_dtype
        ).to(self.device)

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None
        ).to(self.device)

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.controlnet_strength = controlnet_strength

    def preprocess_image(self, image, contrast_factor=1.0, sharpness_factor=1.0):
        """線画強調のための前処理"""
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness_factor)
        return image

    def _extract_lines(self, image: Image.Image, threshold: int = 128) -> Image.Image:
        """二値化で線画を抽出"""
        gray = image.convert("L")
        binary = gray.point(lambda p: 255 if p > threshold else 0)
        return binary

    def _interpolate_lines(self, img1: Image.Image, img2: Image.Image, t: float) -> Image.Image:
        """動的thresholdを用いた線画補間"""
        def lerp(a: float, b: float, t: float) -> float:
            return a + (b - a) * t

        # 時間に応じて A/B の線の強調度を変化
        thresholdA = int(lerp(100, 150, t))  # Aはだんだん薄く
        thresholdB = int(lerp(150, 100, t))  # Bはだんだん濃く

        lines1 = self._extract_lines(img1, thresholdA)
        lines2 = self._extract_lines(img2, thresholdB)

        blended = Image.blend(lines1, lines2, t)
        return blended.convert("RGB")

    def interpolate_images(
        self,
        image_1: Image.Image,
        image_2: Image.Image,
        num_frames: int,
        prompt: str,
        negative_prompt: str = "",
        contrast: float = 1.0,
        sharpness: float = 1.0,
        easing: str = "linear",
        steps: int = 30,
        output_callback=None,
    ):
        """
        線画補間のメイン処理
        - image_1 と image_2 の間を num_frames で補間
        - 各フレームを output_callback(i, result) で返す
        """

        image_1 = self.preprocess_image(image_1.convert("RGB"), contrast, sharpness)
        image_2 = self.preprocess_image(image_2.convert("RGB"), contrast, sharpness)

        easing_fn = EASING_FUNCTIONS.get(easing.lower(), linear)

        for i in range(num_frames):
            t = i / (num_frames - 1)
            eased_t = easing_fn(t)

            if i == 0:
                # 最初はオリジナル線画
                result = image_1
            elif i == num_frames - 1:
                # 最後もオリジナル線画
                result = image_2
            else:
                # 中間は線画補間 → ControlNet
                interpolated_line = self._interpolate_lines(image_1, image_2, eased_t)
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=interpolated_line,
                    controlnet_conditioning_scale=self.controlnet_strength,
                    num_inference_steps=steps
                ).images[0]

            if output_callback:
                output_callback(i, result)

        return True
