# coding: utf-8
"""
스트리밍 추론: MuseTalk 프레임을 생성하면서 실시간으로 FFmpeg에 파이프,
fMP4 청크를 yield해 HTTP 스트리밍에 사용.
"""

import os, queue, threading, subprocess
import numpy as np
import cv2
import torch

from musetalk.utils.blending import get_image_blending
from musetalk.utils.utils import datagen


def inference_stream(
    avatar, audio_path, fps, ffmpeg_bin,
    pe, unet, vae, timesteps,
    whisper_model, audio_processor, weight_dtype, device,
    audio_pad_left=2, audio_pad_right=2,
    taesd=None,
):
    """
    MuseTalk 추론 결과를 프레임 단위로 FFmpeg에 파이프하고
    fMP4 바이너리 청크를 yield하는 제너레이터.

    첫 청크 도달 시간:
      TTS(0.5초) + Whisper(0.04초) + 첫 3배치(~2.7초) + FFmpeg버퍼(~0.5초) ≈ 3~4초
    """
    ffmpeg_exe = os.path.join(ffmpeg_bin, 'ffmpeg.exe')

    # ── 1. 오디오 특징 추출 (빠름: ~40ms) ──────────────────────
    whisper_features, librosa_length = audio_processor.get_audio_feature(
        audio_path, weight_dtype=weight_dtype
    )
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_features, device, weight_dtype, whisper_model, librosa_length,
        fps=fps,
        audio_padding_length_left=audio_pad_left,
        audio_padding_length_right=audio_pad_right,
    )
    video_num = len(whisper_chunks)

    # ── 2. 아바타 프레임 크기 확인 ──────────────────────────────
    H, W = avatar.frame_list_cycle[0].shape[:2]

    # ── 3. FFmpeg 프로세스 시작 ─────────────────────────────────
    cmd = [
        ffmpeg_exe, '-y', '-v', 'quiet',
        '-fflags', '+genpts',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps),
        '-thread_queue_size', '512',
        '-i', 'pipe:0',
        '-thread_queue_size', '512',
        '-i', audio_path,
        '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency', '-crf', '23',
        '-vf', 'format=yuv420p',
        '-g', '12', '-sc_threshold', '0',  # 키프레임 0.48초마다 (12프레임)
        '-c:a', 'aac', '-b:a', '128k',
        '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
        '-shortest', '-f', 'mp4', 'pipe:1',
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    # ── 4. FFmpeg stdout → asyncio Queue 브릿지 스레드 ─────────
    output_q: queue.Queue = queue.Queue()

    def _read_stdout():
        try:
            while True:
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                output_q.put(chunk)
        finally:
            output_q.put(None)

    reader = threading.Thread(target=_read_stdout, daemon=True)
    reader.start()

    # ── 5. GPU 추론과 CPU 블렌딩을 분리해 병렬 실행 ────────────
    # GPU가 다음 배치를 처리하는 동안 CPU는 이전 배치 블렌딩
    blend_q: queue.Queue = queue.Queue(maxsize=64)

    def _decode_fast(pred):
        """VAE 디코딩: 홀수 프레임만 디코딩 후 짝수 프레임은 선형 보간
        - VAE 연산 절반으로 줄임 (~3.5초 단축 예상)
        - 인접 프레임 평균이라 립싱크 품질 영향 최소
        """
        latents = (1 / vae.scaling_factor) * pred
        dtype   = vae.vae.dtype
        n       = latents.shape[0]

        if n >= 4:
            # 키프레임(짝수 인덱스)만 VAE 디코딩
            key_latents = latents[::2]
            key_decoded = vae.vae.decode(key_latents.to(dtype)).sample  # n//2 프레임

            # 벡터화 보간: 인접 키프레임의 평균
            interp = (key_decoded[:-1] + key_decoded[1:]) / 2  # [k-1, C, H, W]

            # 키프레임 + 보간 프레임 인터리브
            k = key_decoded.shape[0]
            pairs = min(k - 1, interp.shape[0])
            merged = torch.stack([key_decoded[:pairs], interp[:pairs]], dim=1)
            merged = merged.reshape(pairs * 2, *key_decoded.shape[1:])
            # 마지막 키프레임 추가 후 정확한 n개로 자름
            image = torch.cat([merged, key_decoded[pairs:]], dim=0)[:n]
        else:
            image = vae.vae.decode(latents.to(dtype)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image * 255).round().to(torch.uint8)
        image = image[:, [2, 1, 0], :, :]
        image = image.permute(0, 2, 3, 1)
        return image.cpu().numpy()

    def _gpu_loop():
        """GPU 전용: 배치 추론 → raw 프레임을 blend_q에 적재"""
        import time as _t
        gen = datagen(whisper_chunks, avatar.input_latent_list_cycle, avatar.batch_size)
        batch_idx = 0
        try:
            for whisper_batch, latent_batch in gen:
                b = whisper_batch.shape[0]
                t0 = _t.perf_counter()
                with torch.inference_mode():
                    af   = pe(whisper_batch.to(device))
                    lb   = latent_batch.to(dtype=unet.model.dtype)
                    pred = unet.model(lb, timesteps, encoder_hidden_states=af).sample
                t1 = _t.perf_counter()
                with torch.inference_mode():
                    recon = _decode_fast(pred)
                t2 = _t.perf_counter()
                print(f"  [배치{batch_idx:02d}] B={b:2d}  UNet={( t1-t0)*1000:5.0f}ms  VAE={(t2-t1)*1000:5.0f}ms  합={(t2-t0)*1000:5.0f}ms", flush=True)
                batch_idx += 1
                for raw in recon:
                    blend_q.put(raw)
        except Exception as e:
            print(f"[stream] GPU 루프 오류: {e}")
        finally:
            blend_q.put(None)

    def _cpu_blend_write():
        """CPU 전용: blend_q에서 raw 프레임 꺼내 블렌딩 → FFmpeg stdin"""
        idx = 0
        try:
            while True:
                raw = blend_q.get()
                if raw is None or idx >= video_num:
                    break
                bbox  = avatar.coord_list_cycle[idx % len(avatar.coord_list_cycle)]
                ori   = avatar.frame_list_cycle[idx % len(avatar.frame_list_cycle)].copy()
                x1, y1, x2, y2 = bbox
                rf    = cv2.resize(raw.astype(np.uint8), (x2 - x1, y2 - y1))
                mask  = avatar.mask_list_cycle[idx % len(avatar.mask_list_cycle)]
                mc    = avatar.mask_coords_list_cycle[idx % len(avatar.mask_coords_list_cycle)]
                frame = get_image_blending(ori, rf, bbox, mask, mc)
                proc.stdin.write(frame.tobytes())
                idx += 1
        except Exception as e:
            print(f"[stream] CPU 블렌딩 오류: {e}")
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    gpu_thread = threading.Thread(target=_gpu_loop, daemon=True)
    writer     = threading.Thread(target=_cpu_blend_write, daemon=True)
    gpu_thread.start()
    writer.start()

    # ── 6. FFmpeg 출력 청크 yield ───────────────────────────────
    while True:
        chunk = output_q.get()
        if chunk is None:
            break
        yield chunk

    writer.join(timeout=5)
    proc.wait(timeout=5)
