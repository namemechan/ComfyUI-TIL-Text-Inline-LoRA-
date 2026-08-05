# ComfyUI-TIL (Text-Inline LoRA)

프롬프트 문자열 안에 LoRA 태그를 직접 적어 넣으면, 별도의 LoRA 로더 노드 없이도
자동으로 파싱해서 모델에 적용해주는 ComfyUI 커스텀 노드입니다.

```
best quality, 1girl, <lora:my_lora:0.8>, solo
```

위와 같은 프롬프트를 넣으면 `my_lora`를 가중치 `0.8`로 적용하고,
태그가 제거된 `best quality, 1girl, solo` 텍스트를 함께 돌려줍니다.

---

## 태그 문법

| 형태 | 설명 |
|---|---|
| `<이름:모델가중치>` | 가장 기본 형태 |
| `<이름:모델가중치:클립가중치>` | 클립 가중치를 따로 지정 |
| `<lora:이름:모델가중치>` | `lora:` 접두사 사용 가능 (대소문자 무관, 생략 가능) |
| `<lora:이름:모델가중치:클립가중치>` | 접두사 + 클립 가중치 |

- `이름`에는 `<`, `>`, `:` 문자를 사용할 수 없습니다.
- 클립가중치를 생략하면 모델가중치와 동일한 값이 사용됩니다.
- 태그는 **등장 순서대로** 순차 적용됩니다.

---

## 노드 구성 (총 5개)

이 팩은 **한 노드가 전부 처리하는 방식(Base)**, **선으로 연결해서 입력과 적용 위치를
떨어뜨리는 방식(Sender → Receiver)**, 그리고 **선 연결 없이 채널 이름으로만 연결하는
방식(Sender (Wireless) → Receiver (Wireless))** 세 가지 사용법을 제공합니다.
다섯 노드 모두 내부적으로 같은 파싱/적용 로직을 공유하므로, 최종 동작 결과는 동일합니다.

### 1. TIL - Base
가장 단순한 형태. 텍스트와 모델을 한 노드에서 함께 받아, 파싱부터 LoRA 적용까지 한 번에 처리합니다.

- **입력**: `model`, `text`, `mode`, `log`, *(선택)* `clip`
- **출력**: `model`, `clip`, `text` (태그 제거된 문자열)
- **언제 쓰나**: 그래프를 간단하게 유지하고 싶을 때, 텍스트 입력 위치와 모델이 가까이 있을 때

### 2. TIL - Sender
텍스트만 입력받아 태그를 파싱합니다. `model` 소켓이 없어서 그래프 어디에나 자유롭게 배치할 수 있습니다.

- **입력**: `text`
- **출력**: `lora_stack` (파싱 결과, `TIL_LORA_STACK` 타입), `text` (태그 제거된 문자열)
- **언제 쓰나**: 프롬프트 작성 노드를 체크포인트 로더와 멀리 떨어뜨려 배치하고 싶을 때

### 3. TIL - Receiver
`TIL - Sender`가 만든 `lora_stack`을 받아 실제로 LoRA 파일을 로드하고 모델에 적용합니다.

- **입력**: `model`, `lora_stack`, `mode`, `log`, *(선택)* `clip`
- **출력**: `model`, `clip`
- **언제 쓰나**: 체크포인트 로더 근처 등 모델이 실제로 흐르는 지점에 배치

> `TIL - Sender` + `TIL - Receiver`를 함께 쓰면, `TIL - Base` 하나를 쓰는 것과 동일한 결과가 나옵니다.

### 4. TIL - Sender (Wireless)
`TIL - Sender`와 동일하게 텍스트를 파싱하지만, 결과를 소켓 연결이 아니라 `channel`이라는
이름의 전역 저장소에 기록합니다. 같은 `channel` 이름을 가진 `TIL - Receiver (Wireless)`가
이 값을 가져다 씁니다. 그래프에 선을 긋지 않아도 되므로 배치가 훨씬 자유롭습니다.

- **입력**: `text`, `channel`, `log`
- **출력**: `text` (태그 제거된 문자열)
- **언제 쓰나**: 그래프가 복잡해서 선을 길게 끌기 번거로울 때, 여러 프롬프트 노드를 하나의
  체크포인트에 이름만으로 매핑하고 싶을 때

### 5. TIL - Receiver (Wireless)
같은 `channel` 이름의 `TIL - Sender (Wireless)`가 저장한 파싱 결과를 가져와 실제로 LoRA
파일을 로드하고 모델에 적용합니다.

- **입력**: `model`, `channel`, `mode`, `log`, *(선택)* `clip`
- **출력**: `model`, `clip`
- **언제 쓰나**: `TIL - Sender (Wireless)`와 짝을 이루어 쓸 때

> ⚠️ **주의: 선 연결이 아니므로 ComfyUI가 실행 순서를 보장하지 않습니다.**
> 같은 실행(run) 안에서 Receiver가 Sender보다 먼저 실행될 수도 있습니다. 이를 위해:
> - 해당 `channel`에 아직 값이 없으면 경고를 남기고 `model`/`clip`을 변형 없이 그대로
>   통과시킵니다 (안전한 폴백).
> - 서버를 재시작하지 않는 한 저장소에는 "직전에 성공한 Sender"의 값이 남아있으므로,
>   그래프에서 Sender를 지우거나 비활성화한 채 Receiver만 실행하면 예전 값을 잘못 적용할
>   위험이 있습니다. 이를 막기 위해 Sender/Receiver 모두 ComfyUI가 넘겨주는 워크플로우
>   그래프(`PROMPT`)를 해시한 "실행 지문"을 함께 저장·비교하여, 지문이 다르면(=이번 실행에
>   속한 값이 아니면) 오래된 값을 무시하고 통과시킵니다.
> - 두 Wireless 노드는 매 실행마다 강제로 다시 돌도록 설정되어 있어(`IS_CHANGED`), 입력이
>   그대로여도 채널 값이 갱신됩니다.

---

## 공통 옵션

| 옵션 | 값 | 설명 |
|---|---|---|
| `mode` | `auto` / `clip` / `no_clip` | `auto`는 LoRA 파일 내부 키를 검사해 CLIP 가중치 포함 여부를 자동 판단합니다. `clip`은 강제로 CLIP까지 적용, `no_clip`은 모델에만 적용합니다. |
| `log` | `True` / `False` | 적용 성공/실패 로그를 콘솔에 출력할지 여부 |
| `channel` | 문자열 (기본값 `"TIL"`) | Wireless 노드 전용. 같은 이름의 Sender (Wireless) ↔ Receiver (Wireless)를 연결하는 식별자 |

- `mode`가 `clip` 또는 `auto`(CLIP 키 포함 판단)이지만 `clip` 입력이 연결되어 있지 않으면,
  경고 로그를 남기고 모델에만 적용합니다.
- 태그에 적힌 이름과 일치하는 LoRA 파일을 찾지 못하면 해당 태그는 건너뛰고 로그를 남깁니다.

---

## 파일 구성

```
comfyui-til/
├── __init__.py      # 노드 매핑 export
└── til_nodes.py      # 태그 파싱/적용 로직 + 노드 3개 정의 (단일 파일 통합)
```

---

## 예시 워크플로우

**단순 구성**
```
CheckpointLoader → TIL - Base (model, text 연결) → CLIP Text Encode → KSampler
```

**분리 구성**
```
(자유 위치)  프롬프트 텍스트 → TIL - Sender → lora_stack ──┐
                                                             ↓
CheckpointLoader → TIL - Receiver (model, lora_stack 연결) → CLIP Text Encode → KSampler
```

**무선(Wireless) 구성**
```
(자유 위치)  프롬프트 텍스트 → TIL - Sender (Wireless) [channel="TIL"]   (선 연결 없음)

CheckpointLoader → TIL - Receiver (Wireless) [channel="TIL"] (model 연결) → CLIP Text Encode → KSampler
```
Sender와 Receiver 사이에 선을 긋지 않고, `channel` 값(예: `"TIL"`)만 똑같이 맞춰주면 됩니다.
