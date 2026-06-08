import os
import json
import torch
from argparse import Namespace

import scripts.realtime_inference as rt
from scripts.realtime_inference import Avatar
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from transformers import WhisperModel

args = Namespace(
    version="v15", extra_margin=10, parsing_mode="jaw",
    left_cheek_width=40, right_cheek_width=40, batch_size=16, fps=25,
    audio_padding_length_left=1, audio_padding_length_right=1,
    skip_save_images=False, result_dir="./results",
)
rt.args = args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[서버] 디바이스: {device}")

# ─── 모델 전역 상태 ───────────────────────────────────────────
vae = unet = pe = timesteps = whisper = audio_processor = weight_dtype = fp = None
avatar_short = avatar_long = custom_avatar = None
taesd_decoder = None  # TAESD 경량 VAE 디코더
CUSTOM_AVATAR_CACHE = f"./results/{args.version}/avatars/custom_avatar"

models_ready        = False
loading_status      = "모델 미로드"
loading_error       = None
loading_in_progress = False

def load_models():
    global vae, unet, pe, timesteps, whisper, audio_processor, weight_dtype, fp
    global avatar_short, avatar_long, models_ready, loading_status, loading_error, loading_in_progress
    try:
        loading_status = "MuseTalk 모델 로딩 중..."
        vae, unet, pe = load_all_model(
            unet_model_path="./models/musetalkV15/unet.pth",
            vae_type="sd-vae",
            unet_config="./models/musetalkV15/musetalk.json",
            device=device,
        )
        timesteps = torch.tensor([0], device=device)
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)
        weight_dtype = unet.model.dtype

        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("[서버] SDPA (Flash Attention) 활성화")
        except Exception as se:
            print(f"[서버] SDPA 스킵: {se}")


        audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        whisper = WhisperModel.from_pretrained("./models/whisper")
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)

        fp = FaceParsing(left_cheek_width=40, right_cheek_width=40)

        rt.vae = vae; rt.unet = unet; rt.pe = pe; rt.timesteps = timesteps
        rt.whisper = whisper; rt.audio_processor = audio_processor
        rt.weight_dtype = weight_dtype; rt.device = device; rt.fp = fp

        loading_status = "아바타 준비 중..."
        avatar_short = Avatar(avatar_id="bartender",      video_path="data/video/bartender.mp4",      bbox_shift=0, batch_size=args.batch_size, preparation=True)
        avatar_long  = Avatar(avatar_id="bartender_long", video_path="data/video/Bartender_long.mp4", bbox_shift=0, batch_size=args.batch_size, preparation=True)

        # latent를 GPU에 사전 적재 (추론 시 CPU→GPU 전송 제거)
        avatar_short.input_latent_list_cycle = [t.to(device) for t in avatar_short.input_latent_list_cycle]
        avatar_long.input_latent_list_cycle  = [t.to(device) for t in avatar_long.input_latent_list_cycle]
        print("[서버] latent GPU 적재 완료")

        _try_load_custom_avatar_from_cache()

        # GPU 웜업: CUDA 커널 사전 로딩으로 첫 추론 콜드 스타트 제거
        _warmup_gpu()

        models_ready   = True
        loading_status = "준비 완료"
        print("[서버] 모델 준비 완료")
    except Exception as e:
        loading_error  = str(e)
        loading_status = f"로딩 실패: {e}"
        print(f"[서버] 로딩 실패: {e}")
    finally:
        loading_in_progress = False

def _try_load_custom_avatar_from_cache():
    global custom_avatar
    info_path = os.path.join(CUSTOM_AVATAR_CACHE, "avator_info.json")
    if not os.path.exists(info_path):
        return
    try:
        with open(info_path) as f:
            info = json.load(f)
        custom_avatar = Avatar(
            avatar_id="custom_avatar",
            video_path=info.get("video_path", ""),
            bbox_shift=info.get("bbox_shift", 0),
            batch_size=args.batch_size,
            preparation=False,
        )
        custom_avatar.input_latent_list_cycle = [t.to(device) for t in custom_avatar.input_latent_list_cycle]
    except Exception as e:
        print(f"[캐시] 커스텀 아바타 로드 실패: {e}")

def _warmup_gpu():
    """UNet+VAE CUDA 커널 사전 컴파일.
    batch=16(정상 배치) + 소배치 4/8/12(마지막 배치 후보)를 모두 실행해
    실제 추론 시 알고리즘 탐색 없이 바로 캐시된 커널을 사용하게 함."""
    print("[서버] GPU 웜업 중...")
    try:
        with torch.inference_mode():
            # batch=16 정상 경로 (frame interpolation과 동일하게 out[::2] 사용)
            _w16 = torch.zeros(16, 5, 384, dtype=weight_dtype, device=device)
            _l16 = torch.zeros(16, 8, 32, 32, dtype=weight_dtype, device=device)
            _o16 = unet.model(_l16, timesteps, encoder_hidden_states=pe(_w16)).sample
            vae.vae.decode(_o16[::2].to(vae.vae.dtype))
            torch.cuda.synchronize()
            print("[서버] 웜업 batch=16 완료")

            # 소배치: 마지막 배치에서 발생하는 CUDA 알고리즘 탐색 제거
            for b in [4, 8, 12]:
                _w = torch.zeros(b, 5, 384, dtype=weight_dtype, device=device)
                _l = torch.zeros(b, 8, 32, 32, dtype=weight_dtype, device=device)
                _o = unet.model(_l, timesteps, encoder_hidden_states=pe(_w)).sample
                key = _o[::2]
                if key.shape[0] >= 1:
                    vae.vae.decode(key.to(vae.vae.dtype))
                torch.cuda.synchronize()
                print(f"[서버] 웜업 batch={b} 완료")

        print("[서버] GPU 웜업 완료")
    except Exception as e:
        print(f"[서버] GPU 웜업 스킵: {e}")

def offload_to_cpu():
    vae.vae = vae.vae.cpu(); unet.model = unet.model.cpu()
    pe.cpu(); whisper.cpu()
    torch.cuda.empty_cache()

def reload_to_gpu():
    vae.vae = vae.vae.half().to(device); unet.model = unet.model.half().to(device)
    pe.half().to(device); whisper.to(device=device, dtype=weight_dtype)
    rt.vae = vae; rt.unet = unet; rt.pe = pe; rt.whisper = whisper
