import os
import base64
from typing import List, Tuple, Dict
import google.generativeai as genai
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY가 .env에 설정되어 있지 않습니다.")

genai.configure(api_key=GOOGLE_API_KEY)

# 기존 시스템 프롬프트(주석처리)
# SYSTEM_PROMPT = "Extract all visible text from the provided image. Output only the text."
# 이전 프롬프트(더 긴 버전)는 위쪽에 이미 주석으로 남아있음

SYSTEM_PROMPT = (
    "Extract only the main body text from the provided image. "
    "For paragraph breaks, do NOT use original line breaks; instead, insert a line break only when a new paragraph starts, as indicated by an indented first character. "
    "Do NOT extract or output page numbers, titles, headers, footers, annotations, or footnotes—only the main body text. "
    "Do NOT extract or output any text from or related to images, tables, charts, diagrams, or figures, including their titles, captions, and any text inside them. "
    "If the page appears to be a book cover, output only the book title. "
    "Output only the main body text, preserving paragraph boundaries as described."
)

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_text_from_image(
    image_paths: List[str],
    model_name: str = "models/gemini-2.5-flash-preview-05-20",
    temperature: float = 0.1
) -> Dict:
    """
    PNG 이미지 리스트(최대 40장)를 받아 Gemini Flash로 텍스트 추출
    """
    import traceback
    if not (1 <= len(image_paths) <= 40):
        return {"text": "", "usage": None, "error": "이미지 개수는 1~40장만 허용됩니다."}

    parts = [
        {"role": "user", "parts": [SYSTEM_PROMPT] + [
            {"inline_data": {"mime_type": "image/png", "data": image_to_base64(img)}} for img in image_paths
        ]}
    ]

    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(
            parts,
            generation_config={"temperature": temperature},
            stream=False
        )
        # 안전하게 텍스트 추출
        text = ""
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts:
                part = candidate.content.parts[0]
                if hasattr(part, "text"):
                    text = part.text
        usage = getattr(response, "usage_metadata", None)
        return {"text": text, "usage": usage, "error": None}
    except Exception as e:
        traceback.print_exc()
        return {"text": "", "usage": None, "error": str(e)}


if __name__ == "__main__":
    # 예시: page_txt/아무_폴더/0001.png, 0002.png, 0003.png 등 테스트
    import glob
    img_dir = input("이미지 폴더 경로를 입력하세요 (예: page_txt/파일명): ").strip()
    all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not all_imgs:
        print("해당 폴더에 PNG 이미지가 없습니다.")
        exit(1)
    print("\n[이미지 파일 목록]")
    for idx, fname in enumerate(all_imgs, 1):
        print(f"{idx}. {os.path.basename(fname)}")
    sel = input("\n변환할 이미지 번호(쉼표로 최대 3개, 예: 1,2,3): ").strip()
    try:
        sel_idx = [int(s)-1 for s in sel.split(",") if s.strip().isdigit()]
        sel_imgs = [all_imgs[i] for i in sel_idx if 0 <= i < len(all_imgs)]
        if not (1 <= len(sel_imgs) <= 3):
            print("1~3개 이미지를 선택해야 합니다.")
            exit(1)
    except Exception:
        print("입력 오류. 다시 시도하세요.")
        exit(1)
    print(f"\nGemini Flash로 텍스트 추출 중... (이미지 {len(sel_imgs)}장)")
    result = extract_text_from_image(sel_imgs)
    print("\n[추출 결과]")
    print(result.get("text", "(결과 없음)"))
    if result.get("usage"):
        print("\n[토큰/비용 정보]")
        print(result["usage"])
    if result.get("error"):
        print(f"\n[오류] {result['error']}") 