"""
comfyui-TIL (Text-Inline LoRA)

문자열(프롬프트) 안의 <이름:가중치> / <lora:이름:모델가중치:클립가중치> 태그를
등장 순서대로 파싱해 model(및 필요 시 clip)에 적용한다.

태그 문법:
    <이름:모델가중치>
    <이름:모델가중치:클립가중치>
    <lora:이름:모델가중치>            ("lora:" / "Lora:" / "LORA:" 등 접두사 선택적 허용)
    <lora:이름:모델가중치:클립가중치>

클립가중치가 생략되면 모델가중치와 동일한 값을 사용한다 (WebUI 관례와 동일).

노드 구성 (6개):
    TIL - Base       : 텍스트 + 모델을 한 노드가 함께 받아 파싱·적용까지 한 번에 처리 (로컬형)
    TIL - Sender      : 텍스트만 파싱해 결과(TIL_LORA_STACK)를 내보냄. model 소켓 없음 (원격 송신측)
    TIL - Receiver     : Sender 가 만든 결과를 받아 model(및 clip)에 실제로 적용 (원격 수신측)
    TIL - Sender (Wireless) / Receiver (Wireless): 채널 문자열로 결과를 전달하고 적용
    TIL - LoRA Checker: 모델 또는 LoRA의 블록 구조를 문자열로 출력
"""

import os
import re
import json
import hashlib
import logging
import math
from collections import Counter

import folder_paths
import comfy.utils
import comfy.sd
import comfy.lora
import comfy.lora_convert

logger = logging.getLogger("TIL")

# <(lora:)이름:모델가중치(:클립가중치)>
# - 접두사 "lora:" 는 대소문자 무관, 선택적
# - 이름에는 <, >, : 를 포함하지 않는다고 가정
TAG_PATTERN = re.compile(
    r"<(?:[Ll][Oo][Rr][Aa]:)?([^<>:]+):(-?\d+(?:\.\d+)?)(?::(-?\d+(?:\.\d+)?))?>"
)

# TAG_PATTERN 매칭에는 실패했지만 태그처럼 "<...>" 형태로 보이는 문자열을
# 잡아내기 위한 느슨한 패턴 (오타/형식 오류 감지용, 실제 파싱에는 사용하지 않음)
LOOSE_TAG_LOOKALIKE_PATTERN = re.compile(r"<[^<>]*>")


# ---------------------------------------------------------------------------
# 공통 로직
# ---------------------------------------------------------------------------
def parse_tags(text: str, log: bool = True) -> list[dict]:
    """
    텍스트에서 태그를 등장 순서대로 추출한다.
    반환 항목: {"name": str, "weight_model": float, "weight_clip": float|None, "raw": str}
    weight_clip 이 None 이면 "생략됨" (모델가중치와 동일하게 취급).

    log=True 이면, "<...>" 형태로 보이지만 TAG_PATTERN 에 매칭되지 않은
    (즉 형식 오류로 무시된) 문자열에 대해 경고 로그를 남긴다.
    """
    tags = []
    for m in TAG_PATTERN.finditer(text):
        name = m.group(1).strip()

        try:
            w_model = float(m.group(2))
        except ValueError:
            continue

        w_clip = None
        if m.group(3) is not None:
            try:
                w_clip = float(m.group(3))
            except ValueError:
                w_clip = None

        tags.append(
            {
                "name": name,
                "weight_model": w_model,
                "weight_clip": w_clip,
                "raw": m.group(0),
            }
        )

    if log:
        _warn_unmatched_tag_lookalikes(text, tags)

    return tags


def _warn_unmatched_tag_lookalikes(text: str, parsed_tags: list[dict]):
    """
    "<...>" 형태이지만 TAG_PATTERN 에 매칭되지 않아 무시된 문자열을 찾아
    경고 로그를 남긴다. (예: 세그먼트가 3개보다 많거나, 숫자 형식이 아닌 경우 등)
    """
    matched_raws = {t["raw"] for t in parsed_tags}
    for m in LOOSE_TAG_LOOKALIKE_PATTERN.finditer(text):
        raw = m.group(0)
        if raw in matched_raws:
            continue
        log_msg(
            True,
            f"[TIL] 경고: 태그 형식으로 보이지만 인식되지 않아 무시됨 (원문='{raw}')",
        )


def clean_text(text: str) -> str:
    """태그를 전부 제거한 텍스트를 반환한다."""
    return TAG_PATTERN.sub("", text)


def find_lora_filename(name: str, available: list[str], log: bool = False) -> str | None:
    """
    태그에 적힌 이름(확장자 없을 수 있음)으로 실제 로라 파일명을 찾는다.
    1) 확장자까지 정확히 일치
    2) 확장자 제외, 대소문자 구분 완전 일치
    3) 확장자 제외, 대소문자 무시 완전 일치 (폴백)
    실패하면 None.

    log=True 이면, 3) 폴백 단계에서 매칭되는 후보가 2개 이상일 때
    (선택되지 않은 후보가 존재할 때) 경고 로그를 남긴다.
    """
    if name in available:
        return name

    for f in available:
        base, _ext = os.path.splitext(f)
        if base == name:
            return f

    lname = name.lower()
    candidates = [f for f in available if os.path.splitext(f)[0].lower() == lname]
    if not candidates:
        return None

    if log and len(candidates) > 1:
        log_msg(
            True,
            f"[TIL] 경고: 이름='{name}' 대소문자 무시 매칭 후보가 {len(candidates)}개 "
            f"있어 '{candidates[0]}'를 선택함 (후보: {candidates})",
        )

    return candidates[0]


def lora_has_clip_keys(lora_sd: dict) -> bool:
    """
    로라 state_dict 키를 보고 CLIP(텍스트 인코더) 관련 가중치가 포함되어 있는지
    휴리스틱으로 판단한다.
    """
    for k in lora_sd.keys():
        lk = k.lower()
        if (
            "lora_te" in lk
            or "text_encoder" in lk
            or "te1_" in lk
            or "te2_" in lk
            or lk.startswith("clip_")
        ):
            return True
    return False


def _is_clip_lora_key(key: str) -> bool:
    lk = key.lower()
    return (
        "lora_te" in lk
        or "text_encoder" in lk
        or "te1_" in lk
        or "te2_" in lk
        or lk.startswith("clip_")
    )


def fingerprint_prompt(prompt) -> str | None:
    """
    ComfyUI가 hidden input "PROMPT"로 넘겨주는 워크플로우 그래프 딕셔너리를
    해시하여 "이번 실행(run)"을 식별하는 지문 문자열을 만든다.

    같은 그래프(같은 실행)에 속한 모든 노드는 동일한 prompt 딕셔너리를
    전달받으므로, Sender/Receiver가 같은 실행에서 함께 돌았는지를
    이 지문값의 일치 여부로 판별할 수 있다 (시간 임계값 불필요).

    prompt 가 없거나(None) 직렬화에 실패하면 None 을 반환한다.
    """
    if not prompt:
        return None
    try:
        serialized = json.dumps(prompt, sort_keys=True, default=str)
    except Exception:
        return None
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# Stable ordering for the common diffusion block naming schemes used by
# ComfyUI's model state_dict keys and LoRA keys. The checker exposes this
# ordering as zero-based indices; it does not attempt to describe block roles.
_BLOCK_FAMILY_ORDER = (
    "input_blocks",
    "down_blocks",
    "double_blocks",
    "joint_blocks",
    "transformer_blocks",
    "blocks",
    "layers",
    "middle_block",
    "mid_block",
    "single_blocks",
    "single_transformer_blocks",
    "output_blocks",
    "up_blocks",
)
_INDEXED_BLOCK_FAMILIES = tuple(
    family for family in _BLOCK_FAMILY_ORDER
    if family not in ("middle_block", "mid_block")
)
_BLOCK_DOT_PATTERNS = [
    (family, re.compile(r"(?:^|\.)" + re.escape(family) + r"\.(\d+)(?:\.|$)"))
    for family in _INDEXED_BLOCK_FAMILIES
]
_BLOCK_UNDERSCORE_PATTERNS = [
    (family, re.compile(r"(?:^|_)" + re.escape(family) + r"_(\d+)(?:_|$)"))
    for family in _INDEXED_BLOCK_FAMILIES
]


def _block_descriptor(key: str) -> tuple[str, int | None] | None:
    """Return the architecture-local block family/index represented by a key."""
    for family in ("middle_block", "mid_block"):
        if re.search(r"(?:^|\.)" + family + r"(?:\.|$)", key):
            return (family, None)

    for patterns in (_BLOCK_DOT_PATTERNS, _BLOCK_UNDERSCORE_PATTERNS):
        for family, pattern in patterns:
            match = pattern.search(key)
            if match:
                return (family, int(match.group(1)))
    return None


def _ordered_block_map(keys) -> dict[tuple[str, int | None], int]:
    """Map discovered block groups to stable zero-based display indices."""
    found = set()
    for key in keys:
        descriptor = _block_descriptor(str(key))
        if descriptor is not None:
            found.add(descriptor)

    order = {family: i for i, family in enumerate(_BLOCK_FAMILY_ORDER)}
    return {
        descriptor: index
        for index, descriptor in enumerate(
            sorted(found, key=lambda item: (order[item[0]], -1 if item[1] is None else item[1]))
        )
    }


def _describe_block(descriptor: tuple[str, int | None]) -> str:
    family, block_number = descriptor
    return family if block_number is None else f"{family}.{block_number}"


def _target_key(value) -> str | None:
    """Extract the model key from a ComfyUI LoRA key-map value."""
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], str):
        return value[0]
    return None


def _target_keys(key_map: dict) -> set[str]:
    return {key for key in (_target_key(value) for value in key_map.values()) if key}


def _block_warning(message: str):
    """Block validation warnings are always visible, independent of the success-log toggle."""
    print(message)
    logger.warning(message)


def _parse_block_weights(value: str, block_map: dict) -> dict[int, float]:
    """Parse comma-separated index:strength entries; omitted blocks stay at 1."""
    result = {}
    if not value:
        return result
    # Whitespace has no meaning in this compact syntax, so normalize it away.
    value = "".join(value.split())
    if not value:
        return result

    for item in value.split(","):
        if not item:
            continue
        if item.count(":") != 1:
            _block_warning(f"[TIL] 경고: 블록 설정 '{item}' 형식이 잘못되어 무시합니다. (예: 0:0.5)")
            continue

        block_text, strength_text = item.split(":", 1)
        if not re.fullmatch(r"\d+", block_text):
            _block_warning(f"[TIL] 경고: 블록 번호 '{block_text}'가 0 이상의 정수가 아니어서 무시합니다.")
            continue

        block_index = int(block_text)
        try:
            strength = float(strength_text)
        except (TypeError, ValueError):
            _block_warning(f"[TIL] 경고: 블록 {block_text}의 강도 '{strength_text}'를 숫자로 읽을 수 없어 무시합니다.")
            continue

        if not math.isfinite(strength):
            _block_warning(f"[TIL] 경고: 블록 {block_text}의 강도는 유한한 숫자여야 하므로 무시합니다.")
            continue
        if block_index >= len(block_map):
            _block_warning(f"[TIL] 경고: 블록 {block_index:02d}은 감지된 범위(00-{max(len(block_map) - 1, 0):02d}) 밖이라 무시합니다.")
            continue
        if strength < 0:
            _block_warning(f"[TIL] 경고: 블록 {block_index:02d}의 강도 {strength}는 0으로 제한합니다.")
            strength = 0.0
        elif strength > 1:
            _block_warning(f"[TIL] 경고: 블록 {block_index:02d}의 강도 {strength}는 1로 제한합니다.")
            strength = 1.0

        if block_index in result:
            _block_warning(f"[TIL] 경고: 블록 {block_index:02d}이 중복 지정되어 마지막 값을 사용합니다.")
        result[block_index] = strength

    return result


def _load_lora_with_block_weights(model, clip, lora_sd, strength_model, strength_clip, block_map, block_weights):
    """Load a LoRA and scale model patches by their discovered block index."""
    model_key_map = {}
    clip_key_map = {}
    if model is not None:
        model_key_map = comfy.lora.model_lora_keys_unet(model.model, model_key_map)
    if clip is not None:
        clip_key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, clip_key_map)

    key_map = dict(model_key_map)
    key_map.update(clip_key_map)
    converted_lora = comfy.lora_convert.convert_lora(lora_sd)
    loaded = comfy.lora.load_lora(converted_lora, key_map)

    model_targets = _target_keys(model_key_map)
    clip_targets = _target_keys(clip_key_map)
    out_model = model.clone() if model is not None else None
    out_clip = clip.clone() if clip is not None else None

    model_groups: dict[float, dict] = {}
    clip_patches = {}
    for key, patch in loaded.items():
        if key in model_targets:
            descriptor = _block_descriptor(key)
            block_index = block_map.get(descriptor) if descriptor is not None else None
            block_scale = block_weights.get(block_index, 1.0)
            model_groups.setdefault(strength_model * block_scale, {})[key] = patch
        if key in clip_targets:
            clip_patches[key] = patch

    applied = set()
    if out_model is not None:
        for patch_strength, patches in model_groups.items():
            if patch_strength != 0:
                applied.update(out_model.add_patches(patches, patch_strength))
            else:
                # A zero multiplier intentionally disables these recognized patches.
                applied.update(patches)
    if out_clip is not None and clip_patches:
        applied.update(out_clip.add_patches(clip_patches, strength_clip))
    for key in loaded:
        if key not in applied:
            logger.warning("[TIL] NOT LOADED: %s", key)

    return out_model, out_clip


def log_msg(enabled: bool, message: str):
    if enabled:
        print(message)
    logger.info(message)


def apply_tags(model, clip, tags: list[dict], mode: str, log: bool, block_weights: str = ""):
    """
    파싱된 태그 목록을 등장 순서대로 model(및 필요 시 clip)에 순차 적용한다.
    mode: "auto" | "clip" | "no_clip"
    반환: (model, clip)
    """
    available = folder_paths.get_filename_list("loras")

    out_model = model
    out_clip = clip

    block_map = {}
    parsed_block_weights = {}
    if block_weights and block_weights.strip():
        detection_error = None
        try:
            model_key_map = comfy.lora.model_lora_keys_unet(model.model, {})
            block_map = _ordered_block_map(_target_keys(model_key_map))
        except Exception as e:
            detection_error = e
        if not block_map:
            if detection_error is None:
                _block_warning("[TIL] 경고: 모델 블록 구조를 감지하지 못해 블록 설정을 적용할 수 없습니다.")
            else:
                _block_warning(f"[TIL] 경고: 모델 블록 정보를 읽지 못해 블록 설정을 적용할 수 없습니다. ({detection_error})")
        else:
            parsed_block_weights = _parse_block_weights(block_weights, block_map)

    name_counts = Counter(tag["name"] for tag in tags)
    for dup_name, count in name_counts.items():
        if count > 1:
            log_msg(
                log,
                f"[TIL] 경고: '{dup_name}' 이름이 {count}번 등장함 (가중치가 순차 누적 적용됨)",
            )

    for tag in tags:
        name = tag["name"]
        w_model = tag["weight_model"]
        w_clip = tag["weight_clip"] if tag["weight_clip"] is not None else w_model

        filename = find_lora_filename(name, available, log=log)
        if filename is None:
            log_msg(log, f"[TIL] 실패: 파일을 찾을 수 없음 (이름='{name}')")
            continue

        lora_path = folder_paths.get_full_path("loras", filename)
        if lora_path is None:
            log_msg(log, f"[TIL] 실패: 경로를 찾을 수 없음 (파일='{filename}')")
            continue

        try:
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        except Exception as e:
            log_msg(log, f"[TIL] 실패: 로드 중 오류 (파일='{filename}') - {e}")
            continue

        if mode == "clip":
            use_clip = True
        elif mode == "no_clip":
            use_clip = False
        else:  # auto
            use_clip = lora_has_clip_keys(lora_sd)

        if use_clip and out_clip is None:
            log_msg(
                log,
                f"[TIL] 경고: '{filename}'은 CLIP 가중치를 포함하지만 CLIP 입력이 "
                f"연결되지 않아 model에만 적용합니다.",
            )
            use_clip = False

        try:
            if parsed_block_weights:
                new_model, new_clip = _load_lora_with_block_weights(
                    out_model,
                    out_clip if use_clip else None,
                    lora_sd,
                    w_model,
                    w_clip,
                    block_map,
                    parsed_block_weights,
                )
                out_model = new_model
                if use_clip:
                    out_clip = new_clip
            elif use_clip:
                new_model, new_clip = comfy.sd.load_lora_for_models(
                    out_model, out_clip, lora_sd, w_model, w_clip
                )
                out_model, out_clip = new_model, new_clip
            else:
                new_model, _ = comfy.sd.load_lora_for_models(
                    out_model, None, lora_sd, w_model, 0.0
                )
                out_model = new_model

            clip_info = f", clip={w_clip}" if use_clip else ""
            log_msg(
                log,
                f"[TIL] 적용 완료: '{filename}' (model={w_model}{clip_info})",
            )
        except Exception as e:
            log_msg(log, f"[TIL] 실패: 적용 중 오류 (파일='{filename}') - {e}")
            continue

    return out_model, out_clip


# ---------------------------------------------------------------------------
# 노드 1) TIL - Base : 모델이 이 노드를 직접 통과하며 태그 파싱 + 적용을 한 번에 처리
# ---------------------------------------------------------------------------
class TIL_Base:
    """
    TIL - Base
    문자열(프롬프트) 안의 <이름:가중치> / <lora:이름:모델가중치:클립가중치> 태그를
    등장 순서대로 파싱해 model(및 필요 시 clip)에 적용하고,
    태그가 제거된 문자열을 함께 출력한다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                "mode": (["auto", "clip", "no_clip"], {"default": "auto"}),
                "log": ("BOOLEAN", {"default": True}),
                "block_weights": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "블록별 강도. 예: 0:0.5, 2:0, 15:0.15 (미지정 블록은 1)",
                    },
                ),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "text")
    FUNCTION = "run"
    CATEGORY = "TIL"

    def run(self, model, text, mode, log, block_weights, clip=None):
        tags = parse_tags(text, log=log)
        out_model, out_clip = apply_tags(model, clip, tags, mode, log, block_weights)
        cleaned = clean_text(text)
        return (out_model, out_clip, cleaned)


# ---------------------------------------------------------------------------
# 노드 2) TIL - Sender : 순수 텍스트 파서. model 소켓 없음.
#    프롬프트 작성 위치에 자유롭게 배치하고, 결과(LORA 정보 + 정리된 텍스트)를
#    TIL - Receiver 로 멀리 떨어진 곳까지 전달한다.
# ---------------------------------------------------------------------------
class TIL_Sender:
    """
    TIL - Sender
    프롬프트 텍스트를 입력받아 <이름:가중치> 태그를 파싱한 결과(TIL_LORA_STACK)와
    태그가 제거된 텍스트를 출력한다. 모델을 직접 다루지 않으므로 그래프 어디에나
    자유롭게 배치할 수 있다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TIL_LORA_STACK", "STRING")
    RETURN_NAMES = ("lora_stack", "text")
    FUNCTION = "run"
    CATEGORY = "TIL"

    def run(self, text):
        tags = parse_tags(text)
        cleaned = clean_text(text)
        return (tags, cleaned)


# ---------------------------------------------------------------------------
# 노드 3) TIL - Receiver : 체크포인트 로더 근처 등 model이 실제로 흐르는 지점에 배치.
#    TIL - Sender 가 만든 TIL_LORA_STACK 을 받아 실제 로드/적용을 수행.
# ---------------------------------------------------------------------------
class TIL_Receiver:
    """
    TIL - Receiver
    TIL - Sender 에서 전달받은 lora_stack 정보를 사용해 실제로 로라 파일을
    로드하고 model(및 필요 시 clip)에 적용한다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_stack": ("TIL_LORA_STACK",),
                "mode": (["auto", "clip", "no_clip"], {"default": "auto"}),
                "log": ("BOOLEAN", {"default": True}),
                "block_weights": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "블록별 강도. 예: 0:0.5, 2:0, 15:0.15 (미지정 블록은 1)",
                    },
                ),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "run"
    CATEGORY = "TIL"

    def run(self, model, lora_stack, mode, log, block_weights, clip=None):
        out_model, out_clip = apply_tags(model, clip, lora_stack, mode, log, block_weights)
        return (out_model, out_clip)


# ---------------------------------------------------------------------------
# 무선(Wireless) 채널 레지스트리
#    선으로 연결하지 않고 "channel" 문자열 이름만 일치시켜 Sender → Receiver 로
#    값을 전달한다. 그래프 상 정식 의존성이 아니므로, ComfyUI가 실행 순서를
#    보장해주지 않는다. 즉 같은 프롬프트(run) 안에서 Receiver가 Sender보다
#    먼저 실행될 가능성이 있다 (특히 그래프가 복잡하거나 병렬 브랜치가 많을 때).
#    이 경우를 대비해:
#      - Sender 실행 시 해당 channel 에 값을 기록한다.
#      - Receiver 실행 시 해당 channel 값이 아직 없으면 경고를 남기고
#        model/clip 을 변형 없이 그대로 통과시킨다 (안전한 폴백).
#
#    [실행(run) 불일치 감지]
#    서버를 재시작하지 않는 한 _WIRELESS_STORE 에는 "직전에 성공적으로 실행된
#    Sender" 의 값이 계속 남아있다. 만약 그래프에서 Sender 를 지우거나
#    비활성화한 채 Receiver 만 실행하면, Receiver 는 이를 모르고 예전 값을
#    그대로 적용해버릴 수 있다.
#    이를 막기 위해 Sender/Receiver 모두 hidden input "PROMPT"(ComfyUI가
#    실행마다 넘겨주는 전체 워크플로우 그래프 딕셔너리)를 해시한 지문
#    (fingerprint_prompt)을 함께 저장/비교한다. 같은 실행에 속한 모든 노드는
#    동일한 PROMPT 객체를 전달받으므로, 지문이 정확히 일치할 때만 유효한
#    값으로 취급하고, 불일치 시(=이번 실행에 Sender 가 없었거나 그래프가
#    바뀐 경우) 경고를 남기고 model/clip 을 그대로 통과시킨다.
# ---------------------------------------------------------------------------
_WIRELESS_STORE: dict[str, dict] = {}
# 채널별 저장 형식: {"tags": list[dict], "fingerprint": str | None}


class TIL_Sender_Wireless:
    """
    TIL - Sender (Wireless)
    프롬프트 텍스트를 입력받아 태그를 파싱하고, 결과를 소켓 연결 없이
    "channel" 이름으로 전역 저장한다. TIL - Receiver (Wireless) 가 같은
    channel 이름으로 값을 가져간다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                "channel": (
                    "STRING",
                    {
                        "default": "TIL",
                        "tooltip": "이 이름과 동일한 channel 을 가진 Receiver (Wireless) 로 전달됩니다.",
                    },
                ),
                "log": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "TIL"

    @classmethod
    def IS_CHANGED(cls, text, channel, log, prompt=None):
        # 항상 재실행되도록 강제한다. 안 그러면 ComfyUI가 입력이 그대로일 때
        # 함수를 스킵해버려서, 이 노드가 채널에 값을 다시 기록하지 않는다.
        return float("nan")

    def run(self, text, channel, log, prompt=None):
        tags = parse_tags(text, log=log)
        fp = fingerprint_prompt(prompt)
        _WIRELESS_STORE[channel] = {"tags": tags, "fingerprint": fp}
        cleaned = clean_text(text)
        log_msg(
            log,
            f"[TIL][Wireless] 송신 완료: channel='{channel}' (태그 {len(tags)}개)",
        )
        return (cleaned,)


class TIL_Receiver_Wireless:
    """
    TIL - Receiver (Wireless)
    같은 "channel" 이름을 가진 TIL - Sender (Wireless) 가 저장한 태그 정보를
    가져와 실제로 로라 파일을 로드하고 model(및 필요 시 clip)에 적용한다.
    선 연결이 없으므로 실행 순서가 보장되지 않는다. 해당 channel 에 아직
    값이 없으면 경고를 남기고 model/clip 을 그대로 통과시킨다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "channel": (
                    "STRING",
                    {
                        "default": "TIL",
                        "tooltip": "같은 이름의 Sender (Wireless) 로부터 값을 받아옵니다.",
                    },
                ),
                "mode": (["auto", "clip", "no_clip"], {"default": "auto"}),
                "log": ("BOOLEAN", {"default": True}),
                "block_weights": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "블록별 강도. 예: 0:0.5, 2:0, 15:0.15 (미지정 블록은 1)",
                    },
                ),
            },
            "optional": {
                "clip": ("CLIP",),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "run"
    CATEGORY = "TIL"

    @classmethod
    def IS_CHANGED(cls, model, channel, mode, log, block_weights, clip=None, prompt=None):
        # 항상 재실행되도록 강제한다. 안 그러면 ComfyUI가 입력이 그대로일 때
        # 함수를 스킵해버려서, 채널 값이 바뀌어도 반영되지 않는다.
        return float("nan")

    def run(self, model, channel, mode, log, block_weights, clip=None, prompt=None):
        entry = _WIRELESS_STORE.get(channel)

        if entry is None:
            log_msg(
                log,
                f"[TIL][Wireless] 경고: channel='{channel}' 에 대한 송신 데이터가 "
                f"없습니다. 같은 이름의 Sender (Wireless) 노드가 존재하고 먼저 "
                f"실행되었는지 확인하세요. model/clip 은 변경 없이 통과합니다.",
            )
            return (model, clip)

        my_fp = fingerprint_prompt(prompt)
        stored_fp = entry.get("fingerprint")

        # 지문 계산이 가능한 상태(둘 다 not None)인데 값이 다르면,
        # 지금 이 실행에 속한 Sender 가 아니라 예전(이전 실행)에 남은 값이라는
        # 뜻이므로 적용을 거부하고 안전하게 통과시킨다.
        if my_fp is not None and stored_fp is not None and my_fp != stored_fp:
            log_msg(
                log,
                f"[TIL][Wireless] 경고: channel='{channel}' 의 송신 데이터가 "
                f"이번 실행의 것이 아닙니다 (그래프에서 Sender 가 제거/변경되었을 "
                f"가능성). 오래된 값을 무시하고 model/clip 은 변경 없이 "
                f"통과시킵니다.",
            )
            return (model, clip)

        if my_fp is None or stored_fp is None:
            # PROMPT 를 얻지 못한 예외적 상황(예: 구버전 ComfyUI). 지문 비교를
            # 할 수 없으므로 기존 동작(경고 없이 그대로 적용)으로 폴백한다.
            log_msg(
                log,
                f"[TIL][Wireless] 참고: channel='{channel}' 실행 지문을 확인할 "
                f"수 없어 검증 없이 적용합니다.",
            )

        tags = entry.get("tags", [])
        out_model, out_clip = apply_tags(model, clip, tags, mode, log, block_weights)
        return (out_model, out_clip)


class TIL_LoRA_Checker:
    """Report detected model/LoRA block counts and the checker-assigned indices."""

    NO_LORA = "(모델만 검사)"

    @classmethod
    def INPUT_TYPES(cls):
        lora_choices = [cls.NO_LORA] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (lora_choices, {"default": cls.NO_LORA}),
            },
            "optional": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "inspect"
    CATEGORY = "TIL"

    def inspect(self, lora_name, model=None):
        lines = ["TIL LoRA / Model Block Check"]
        model_key_map = {}
        model_block_map = {}

        if model is not None:
            try:
                model_key_map = comfy.lora.model_lora_keys_unet(model.model, {})
                model_block_map = _ordered_block_map(_target_keys(model_key_map))
                model_type = type(model.model).__name__
                model_config = getattr(model.model, "model_config", None)
                config_name = type(model_config).__name__ if model_config is not None else "unknown config"
                lines.append(f"Model type: {model_type} ({config_name})")
                lines.append(f"Mapped model keys: {len(_target_keys(model_key_map))}")
                lines.append(f"Model block groups detected: {len(model_block_map)}")
                if model_block_map:
                    model_keys_by_block = {index: [] for index in model_block_map.values()}
                    for key in _target_keys(model_key_map):
                        descriptor = _block_descriptor(key)
                        if descriptor in model_block_map:
                            model_keys_by_block[model_block_map[descriptor]].append(key)
                    for descriptor, index in sorted(model_block_map.items(), key=lambda item: item[1]):
                        keys = model_keys_by_block[index]
                        lines.append(f"  {index:02d}: {_describe_block(descriptor)} ({len(keys)} mapped model keys)")
                        for key in sorted(keys)[:3]:
                            lines.append(f"      {key}")
                        if len(keys) > 3:
                            lines.append(f"      ... and {len(keys) - 3} more")
                else:
                    lines.append("  No supported block naming pattern was detected in the model.")
            except Exception as e:
                lines.append(f"Model inspection failed: {e}")
        else:
            lines.append("Model: not connected")

        if lora_name != self.NO_LORA:
            try:
                lora_path = folder_paths.get_full_path("loras", lora_name)
                if lora_path is None:
                    lines.append(f"LoRA: file path not found for '{lora_name}'")
                else:
                    lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    lines.append(f"LoRA: {lora_name}")
                    lines.append(f"Path: {lora_path}")
                    lines.append(f"LoRA tensor keys: {len(lora_sd)}")
                    clip_lora_keys = [key for key in lora_sd if _is_clip_lora_key(str(key))]
                    lines.append(f"CLIP / text-encoder keys (not model blocks): {len(clip_lora_keys)}")
                    tensor_values = [value for value in lora_sd.values() if hasattr(value, "numel")]
                    total_params = sum(int(value.numel()) for value in tensor_values)
                    dtype_counts = Counter(str(value.dtype) for value in tensor_values)
                    lines.append(f"Tensor entries: {len(tensor_values)}; total parameters: {total_params:,}")
                    if dtype_counts:
                        lines.append("Dtypes: " + ", ".join(f"{dtype} ({count})" for dtype, count in sorted(dtype_counts.items())))

                    if model is not None and model_key_map:
                        converted_lora = comfy.lora_convert.convert_lora(lora_sd)
                        mapped_lora = comfy.lora.load_lora(converted_lora, model_key_map)
                        mapped_by_block = {index: [] for index in model_block_map.values()}
                        for key in mapped_lora:
                            descriptor = _block_descriptor(key)
                            if descriptor in model_block_map:
                                mapped_by_block[model_block_map[descriptor]].append(key)
                        covered = {index for index, keys in mapped_by_block.items() if keys}
                        lines.append(
                            f"LoRA blocks matched to this model: {len(covered)} / {len(model_block_map)}"
                        )
                        lines.append(
                            "  Matched indices: "
                            + (", ".join(f"{index:02d}" for index in sorted(covered)) or "none")
                        )
                        for index in sorted(covered):
                            keys = sorted(mapped_by_block[index])
                            lines.append(f"  {index:02d}: {len(keys)} mapped LoRA patch keys")
                            for key in keys[:3]:
                                lines.append(f"      {key}")
                            if len(keys) > 3:
                                lines.append(f"      ... and {len(keys) - 3} more")
                    else:
                        model_lora_keys = [
                            key for key in lora_sd if not _is_clip_lora_key(str(key))
                        ]
                        lora_block_map = _ordered_block_map(model_lora_keys)
                        lines.append(
                            f"LoRA block groups estimated from keys: {len(lora_block_map)}"
                        )
                        if model is None:
                            lines.append(
                                "  No model connected; LoRA-only block numbering is an estimate and may differ from the full model."
                            )
                        if lora_block_map:
                            for descriptor, index in sorted(lora_block_map.items(), key=lambda item: item[1]):
                                group_keys = [
                                    key for key in model_lora_keys
                                    if _block_descriptor(str(key)) == descriptor
                                ]
                                group_params = sum(
                                    int(lora_sd[key].numel())
                                    for key in group_keys
                                    if hasattr(lora_sd[key], "numel")
                                )
                                lines.append(
                                    f"  {index:02d}: {_describe_block(descriptor)} "
                                    f"({len(group_keys)} tensor keys, {group_params:,} parameters)"
                                )
                                for key in sorted(group_keys)[:3]:
                                    value = lora_sd[key]
                                    shape = tuple(value.shape) if hasattr(value, "shape") else "unknown"
                                    lines.append(f"      {key} shape={shape}")
                                if len(group_keys) > 3:
                                    lines.append(f"      ... and {len(group_keys) - 3} more")
                        else:
                            lines.append("  No supported block naming pattern was detected in the LoRA keys.")
            except Exception as e:
                lines.append(f"LoRA inspection failed for '{lora_name}': {e}")
        elif model is None:
            lines.append("Connect a model or select a LoRA to inspect.")

        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "TIL_Base": TIL_Base,
    "TIL_Sender": TIL_Sender,
    "TIL_Receiver": TIL_Receiver,
    "TIL_Sender_Wireless": TIL_Sender_Wireless,
    "TIL_Receiver_Wireless": TIL_Receiver_Wireless,
    "TIL_LoRA_Checker": TIL_LoRA_Checker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TIL_Base": "TIL - Base",
    "TIL_Sender": "TIL - Sender",
    "TIL_Receiver": "TIL - Receiver",
    "TIL_Sender_Wireless": "TIL - Sender (Wireless)",
    "TIL_Receiver_Wireless": "TIL - Receiver (Wireless)",
    "TIL_LoRA_Checker": "TIL - LoRA Checker",
}
