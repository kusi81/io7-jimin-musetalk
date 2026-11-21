"""
RunPod Serverless MuseTalk 1.5 Real-Time Lip-Sync Worker for IO7 Sales (Kim Ji-min Gwazang).

Supported actions:
  1. {"action": "warmup"}
     Loads MuseTalk 1.5 UNet/VAE/Whisper & pre-baked Kim Ji-min avatar latents into GPU VRAM.
     Called automatically when a customer joins the IO7 Sales meeting room to hide cold start.
  2. {"action": "lipsync_chunk", "gesture_id": "explain_hands", "frame_offset": 0,
      "audio_pcm24k_b64": "...", "fps": 25, "output_mode": "face_patch"}
     Synthesizes lip-synced 256x256 face patches (WebP/JPEG + 1920x1080 bbox coordinates)
     so the browser composites tack-sharp lip-sync directly onto Kim Ji-min's native 1080p video!
  3. {"action": "health"}
     Lightweight status check.
"""

import base64
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import runpod
import soundfile as sf
import torch

MUSETALK_DIR = Path(os.environ.get("MUSETALK_DIR", "/app/MuseTalk"))
PREBAKED_DIR = Path(os.environ.get("PREBAKED_DIR", "/app/prebaked"))
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", "/app/assets"))
DEFAULT_FPS = int(os.environ.get("LIPSYNC_FPS", "25"))
PATCH_QUALITY = int(os.environ.get("LIPSYNC_PATCH_QUALITY", "82"))
PATCH_FORMAT = os.environ.get("LIPSYNC_PATCH_FORMAT", "webp").lower()  # webp or jpeg

if str(MUSETALK_DIR) not in sys.path:
    sys.path.insert(0, str(MUSETALK_DIR))


class MuseTalkEngine:
    """Singleton GPU engine holding MuseTalk 1.5 weights & Kim Ji-min pre-baked latents in VRAM."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.loaded = False
        self.musetalk_ready = False
        self.load_error = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.audio_processor = None
        self.whisper = None
        self.timesteps = None
        self.clips = {}  # clip_id -> {meta, crops_bgr, latents}

    def ensure_loaded(self):
        if self.loaded:
            return
        t0 = time.perf_counter()

        # 1. Load pre-baked Kim Ji-min 1080p face crops & bounding boxes
        self._load_prebaked_clips()

        # 2. Load MuseTalk 1.5 models into GPU VRAM
        try:
            os.chdir(str(MUSETALK_DIR))
            from musetalk.utils.utils import load_all_model
            from musetalk.utils.audio_processor import AudioProcessor
            from transformers import WhisperModel

            unet_config = "./models/musetalkV15/musetalk.json"
            unet_weights = "./models/musetalkV15/unet.pth"
            whisper_dir = "./models/whisper"

            self.vae, self.unet, self.pe = load_all_model(
                unet_model_path=unet_weights,
                vae_type="sd-vae",
                unet_config=unet_config,
                device=self.device,
            )
            self.pe = self.pe.to(self.device, dtype=self.dtype)
            self.vae.vae = self.vae.vae.to(self.device, dtype=self.dtype)
            self.unet.model = self.unet.model.to(self.device, dtype=self.dtype)
            self.timesteps = torch.tensor([0], device=self.device)

            self.audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
            self.whisper = WhisperModel.from_pretrained(whisper_dir).to(
                device=self.device, dtype=self.dtype
            ).eval()
            self.whisper.requires_grad_(False)

            # Encode all pre-baked Kim Ji-min crops into VAE latents ONCE at warmup!
            self._encode_clip_latents()
            self.musetalk_ready = True
            print(
                f"[MuseTalkEngine] Loaded MuseTalk 1.5 + {len(self.clips)} pre-baked clips "
                f"on {self.device} in {(time.perf_counter() - t0):.2f}s"
            )
        except Exception as exc:
            self.load_error = str(exc)
            self.musetalk_ready = False
            print(f"[MuseTalkEngine] Warning: MuseTalk 1.5 weights load fallback ({exc})")

        self.loaded = True

    def _load_prebaked_clips(self):
        if not PREBAKED_DIR.exists():
            return
        for clip_dir in sorted(PREBAKED_DIR.iterdir()):
            meta_file = clip_dir / "meta.json"
            if not meta_file.exists():
                continue
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            crops_dir = clip_dir / "crops"
            crops = []
            for idx in range(meta["frameCount"]):
                img = cv2.imread(str(crops_dir / f"{idx:04d}.jpg"))
                if img is not None:
                    crops.append(img)
            if crops:
                self.clips[meta["clipId"]] = {
                    "meta": meta,
                    "crops": crops,
                    "latents": None,
                }

    @torch.no_grad()
    def _encode_clip_latents(self):
        """Pre-encode 256x256 face crops to VAE latents once per worker lifetime."""
        for clip_id, bundle in self.clips.items():
            latents = []
            for crop_bgr in bundle["crops"]:
                # MuseTalk expects 256x256 RGB image; VAE returns masked+ref concatenated latent
                latent = self.vae.get_latents_for_unet(crop_bgr)
                latents.append(latent.to(self.device, dtype=self.dtype))
            bundle["latents"] = latents
            print(f"[MuseTalkEngine] Encoded {len(latents)} VRAM latents for '{clip_id}'")

    def _decode_pcm24k_to_wav(self, audio_pcm24k_b64: str, sample_rate: int = 24000):
        raw = base64.b64decode(audio_pcm24k_b64)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(samples) / float(sample_rate)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, samples, sample_rate, subtype="PCM_16")
        tmp.close()
        return tmp.name, samples, duration

    def _encode_patch(self, img_bgr: np.ndarray) -> str:
        if PATCH_FORMAT == "webp":
            ok, buf = cv2.imencode(".webp", img_bgr, [int(cv2.IMWRITE_WEBP_QUALITY), PATCH_QUALITY])
            if ok:
                return base64.b64encode(buf.tobytes()).decode("ascii")
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), PATCH_QUALITY])
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @torch.no_grad()
    def synthesize_chunk(
        self,
        gesture_id: str,
        frame_offset: int,
        audio_pcm24k_b64: str,
        sample_rate: int = 24000,
        fps: int = DEFAULT_FPS,
        batch_size: int = 8,
    ):
        self.ensure_loaded()
        t0 = time.perf_counter()

        clip_id = gesture_id if gesture_id in self.clips else next(iter(self.clips.keys()), None)
        if not clip_id:
            return {"error": "No pre-baked Kim Ji-min clips found in /app/prebaked"}

        bundle = self.clips[clip_id]
        meta = bundle["meta"]
        total_frames = len(bundle["crops"])
        wav_path, samples, duration = self._decode_pcm24k_to_wav(audio_pcm24k_b64, sample_rate)
        num_frames = max(1, int(round(duration * fps)))

        try:
            frames_out = []
            if self.musetalk_ready and bundle["latents"]:
                # Extract Whisper audio features aligned to 25fps
                whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(wav_path)
                whisper_chunks = self.audio_processor.get_whisper_chunk(
                    whisper_input_features,
                    self.device,
                    self.dtype,
                    self.whisper,
                    librosa_length,
                    fps=fps,
                    audio_padding_length_left=2,
                    audio_padding_length_right=2,
                )
                num_frames = min(len(whisper_chunks), max(1, num_frames))

                for start in range(0, num_frames, batch_size):
                    end = min(num_frames, start + batch_size)
                    whisper_batch = torch.stack(
                        [whisper_chunks[i] for i in range(start, end)], dim=0
                    ).to(self.device, dtype=self.dtype)
                    latent_indices = [(frame_offset + i) % total_frames for i in range(start, end)]
                    latent_batch = torch.cat(
                        [bundle["latents"][idx] for idx in latent_indices], dim=0
                    ).to(self.device, dtype=self.dtype)

                    audio_feature_batch = self.pe(whisper_batch)
                    pred_latents = self.unet.model(
                        latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch
                    ).sample
                    recon = self.vae.decode_latents(pred_latents)

                    for local_i, res_face in enumerate(recon):
                        f_idx = latent_indices[local_i]
                        frames_out.append(
                            {
                                "frameIdx": f_idx,
                                "bbox": meta["bboxes"][f_idx],
                                "patchB64": self._encode_patch(res_face),
                            }
                        )
            else:
                # Diagnostic fallback when running CPU smoke test without MuseTalk weights
                for i in range(num_frames):
                    f_idx = (frame_offset + i) % total_frames
                    frames_out.append(
                        {
                            "frameIdx": f_idx,
                            "bbox": meta["bboxes"][f_idx],
                            "patchB64": self._encode_patch(bundle["crops"][f_idx]),
                        }
                    )

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            return {
                "ok": True,
                "clipId": clip_id,
                "fps": fps,
                "refWidth": meta["width"],
                "refHeight": meta["height"],
                "patchMime": "image/webp" if PATCH_FORMAT == "webp" else "image/jpeg",
                "durationSec": round(duration, 4),
                "nextFrameOffset": (frame_offset + len(frames_out)) % total_frames,
                "inferenceMs": elapsed_ms,
                "musetalkActive": self.musetalk_ready,
                "frames": frames_out,
            }
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)


ENGINE = MuseTalkEngine()


def handler(job):
    inp = job.get("input") or {}
    action = inp.get("action", "lipsync_chunk")

    if action == "health":
        return {
            "ok": True,
            "loaded": ENGINE.loaded,
            "musetalkReady": ENGINE.musetalk_ready,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "clips": list(ENGINE.clips.keys()),
        }

    if action == "warmup":
        t0 = time.perf_counter()
        ENGINE.ensure_loaded()
        return {
            "ok": True,
            "warm": True,
            "musetalkReady": ENGINE.musetalk_ready,
            "loadError": ENGINE.load_error,
            "warmupMs": round((time.perf_counter() - t0) * 1000, 1),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "clips": {cid: len(b["crops"]) for cid, b in ENGINE.clips.items()},
        }

    if action == "lipsync_chunk":
        audio_b64 = inp.get("audio_pcm24k_b64")
        if not audio_b64:
            return {"ok": False, "error": "Missing audio_pcm24k_b64"}
        return ENGINE.synthesize_chunk(
            gesture_id=inp.get("gesture_id", "explain_hands"),
            frame_offset=int(inp.get("frame_offset", 0)),
            audio_pcm24k_b64=audio_b64,
            sample_rate=int(inp.get("sample_rate", 24000)),
            fps=int(inp.get("fps", DEFAULT_FPS)),
            batch_size=int(inp.get("batch_size", 8)),
        )

    return {"ok": False, "error": f"Unknown action: {action}"}


if __name__ == "__main__":
    # Eagerly pre-load on container start so FlashBoot captures warm GPU state
    if os.environ.get("EAGER_WARMUP", "1") == "1":
        ENGINE.ensure_loaded()
    runpod.serverless.start({"handler": handler})
