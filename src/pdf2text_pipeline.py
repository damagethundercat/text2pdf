import os
import sys
import glob
from typing import List, Optional
from tqdm import tqdm
import argparse
from abc import ABC, abstractmethod
import requests
import base64
import subprocess
import shutil
import socket
from pypdf import PdfReader, PdfWriter
import re

sys.path.append(os.path.join(os.path.dirname(__file__)))
from pdf_to_png import convert_pdf_to_images
from window_loader import get_page_windows
from extract_text_flash import extract_text_from_image, SYSTEM_PROMPT
from text_postprocess import clean_text, gemini_paragraph_postprocess

# === ModelHandler 인터페이스 및 팩토리 ===
class ModelHandler(ABC):
    @abstractmethod
    def extract_text(self, image_paths):
        pass

class GeminiModelHandler(ModelHandler):
    def extract_text(self, image_paths):
        # 기존 extract_text_from_image 함수 사용
        return extract_text_from_image(image_paths)

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", model_name="gemma3:12b"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def extract_text(self, image_paths):
        # 이미지 base64 인코딩
        images_b64 = []
        for img_path in image_paths:
            with open(img_path, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode("utf-8"))
        # 프롬프트(임시)
        prompt = (
            "Extract only the main body text from the provided images. "
            "Do NOT extract or output any text from images, tables, charts, diagrams, figures, captions, footnotes, or annotations. "
            "Ignore all visual elements and their associated text. Output only the main body paragraphs as plain text."
        )
        # Ollama API 요청
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": images_b64,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        import time
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("response", "").strip()
                    return {"text": text, "usage": {}}
                elif resp.status_code == 429:
                    print(f"[WARN] Ollama 429 Too Many Requests, 60초 대기 후 재시도 ({attempt+1}/3)")
                    time.sleep(60)
                else:
                    print(f"[ERROR] Ollama API 오류: {resp.status_code} {resp.text}")
                    time.sleep(2)
            except Exception as e:
                print(f"[ERROR] Ollama API 호출 실패: {e}")
                time.sleep(2)
        print("[ERROR] Ollama API 3회 시도 모두 실패. 빈 문자열 반환")
        return {"text": "", "usage": {}, "error": "Ollama API 호출 실패"}

# OllamaGemmaModelHandler에서 OllamaClient 사용
class OllamaGemmaModelHandler(ModelHandler):
    def __init__(self):
        self.client = OllamaClient()
    def extract_text(self, image_paths):
        return self.client.extract_text(image_paths)

# ProVisionModelHandler → Gemini25ProModelHandler로 명칭 변경
class Gemini25ProModelHandler(ModelHandler):
    def extract_text(self, pdf_paths, prompt=SYSTEM_PROMPT):
        results = []
        for pdf_path in pdf_paths:
            result = extract_text_from_pdf(pdf_path, prompt=prompt)
            if result is None:
                result = {"text": "", "usage": {}, "error": "extract_text_from_pdf returned None"}
            results.append(result.get("text", ""))
        return {"text": "\n\n*****\n\n".join(results), "usage": {}}

class ModelHandlerFactory:
    @staticmethod
    def get_handler(model: str) -> ModelHandler:
        if model == "gemini":
            return GeminiModelHandler()
        elif model == "ollama-gemma":
            return OllamaGemmaModelHandler()
        elif model == "gemini-2.5-pro":
            return Gemini25ProModelHandler()
        else:
            raise ValueError(f"지원하지 않는 모델: {model}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_books(books_root="books"):
    return [d for d in os.listdir(books_root) if os.path.isdir(os.path.join(books_root, d))]

def get_pdf_path(book_dir, book_name):
    return os.path.join(book_dir, f"{book_name}.pdf")

def save_checkpoint(ckpt_dir, idx, text):
    ensure_dir(ckpt_dir)
    with open(os.path.join(ckpt_dir, f"{idx+1:04d}.txt"), "w", encoding="utf-8") as f:
        f.write(text)

def load_checkpoints(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return set()
    return set(os.path.splitext(f)[0] for f in os.listdir(ckpt_dir) if f.endswith('.txt'))

def merge_checkpoints(ckpt_dir, out_path):
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.txt")))
    merged = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            merged.append(f.read().strip())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n*****\n\n".join(merged))
    #print(f"[병합 완료] {out_path}")

def is_port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect((host, port))
            return True
        except Exception:
            return False

def ensure_ollama_server():
    # ollama 실행파일 확인
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        print("[오류] ollama 실행파일(ollama.exe)이 PATH에 없습니다. https://ollama.com/download에서 설치 후 재시도하세요.")
        exit(1)
    # 서버 기동 확인
    if is_port_open("localhost", 11434):
        print("[안내] Ollama 서버가 이미 실행 중입니다.")
        return
    print("[안내] Ollama 서버를 자동 실행합니다...")
    try:
        # 윈도우: 새 콘솔에서 백그라운드 실행
        subprocess.Popen([ollama_path, "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    except Exception as e:
        print(f"[오류] Ollama 서버 자동 실행 실패: {e}")
        exit(1)
    # 서버 기동 대기
    import time
    for i in range(30):
        if is_port_open("localhost", 11434):
            print("[안내] Ollama 서버가 정상적으로 실행되었습니다.")
            return
        time.sleep(1)
    print("[오류] Ollama 서버가 30초 내에 실행되지 않았습니다. 수동으로 ollama serve를 실행 후 재시도하세요.")
    exit(1)

def ensure_gemma_model():
    import requests
    # gemma3:12b 모델 존재 확인
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        if resp.status_code == 200:
            tags = resp.json().get("models", [])
            if any("gemma3:12b" in m.get("name", "") for m in tags):
                print("[안내] gemma3:12b 모델이 이미 설치되어 있습니다.")
                return
    except Exception:
        pass
    print("[안내] gemma3:12b 모델을 다운로드(pull)합니다. 네트워크 환경에 따라 수 분 소요될 수 있습니다...")
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        print("[오류] ollama 실행파일(ollama.exe)이 PATH에 없습니다. https://ollama.com/download에서 설치 후 재시도하세요.")
        exit(1)
    try:
        result = subprocess.run([ollama_path, "pull", "gemma3:12b"], check=True)
        print("[안내] gemma3:12b 모델 다운로드가 완료되었습니다.")
    except Exception as e:
        print(f"[오류] gemma3:12b 모델 다운로드(pull) 실패: {e}")
        exit(1)

# 시스템 프롬프트(청크 단위 안내 추가)


def split_pdf(input_pdf, pages_per_chunk=30):
    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)
    chunks = []
    for i in range(0, total_pages, pages_per_chunk):
        writer = PdfWriter()
        for j in range(i, min(i+pages_per_chunk, total_pages)):
            writer.add_page(reader.pages[j])
        chunk_path = f"{os.path.splitext(input_pdf)[0]}_part{i//pages_per_chunk+1}.pdf"
        with open(chunk_path, "wb") as f:
            writer.write(f)
        chunks.append(chunk_path)
    return chunks

# Gemini Pro Vision API PDF 직접 입력 함수(스텁)
def extract_text_from_pdf(pdf_path, model_name="gemini-2.5-pro", prompt=SYSTEM_PROMPT):
    import requests
    import time
    import os
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("[ERROR] GOOGLE_API_KEY가 .env에 설정되어 있지 않습니다.")
        return {"text": "", "usage": {}, "error": "GOOGLE_API_KEY not set"}
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={GOOGLE_API_KEY}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "application/pdf", "data": pdf_b64}}
                    ]
                }
            ]
        }
        resp = requests.post(url, headers=headers, json=data, timeout=600)
        if resp.status_code == 200:
            resp_json = resp.json()
            candidates = resp_json.get("candidates", [])
            if not candidates:
                print("[WARN] Gemini 2.5 Pro 빈 응답(candidates 없음)")
                return {"text": "", "usage": {}, "error": "No candidates"}
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            if not text:
                print("[WARN] Gemini 2.5 Pro 응답에 텍스트 없음")
                return {"text": "", "usage": {}, "error": "No text in response"}
            return {"text": text, "usage": {}}
        elif resp.status_code == 429:
            print(f"[WARN] Gemini 2.5 Pro 429 Too Many Requests")
            return {"text": "", "usage": {}, "error": "429 Too Many Requests"}
        else:
            print(f"[ERROR] Gemini 2.5 Pro API 오류: {resp.status_code} {resp.text}")
            return {"text": "", "usage": {}, "error": f"API error {resp.status_code}"}
    except Exception as e:
        print(f"[ERROR] Gemini 2.5 Pro API 호출 실패: {e}")
        return {"text": "", "usage": {}, "error": str(e)}

PDF_SYSTEM_PROMPT = (
    "You are a professional proof-reader. "
    "Extract all readable text from the provided PDF file and preserve paragraph structure. "
    "When extracting text, DO NOT insert line breaks according to the line breaks in the PDF. "
    "Only insert a line break at the end of a paragraph. "
    "If a paragraph is split across lines or pages, reconstruct it as a single paragraph with no line breaks until the paragraph ends. "
    "If a sentence or paragraph is cut off at the end of a page and continues on the next page, you MUST merge them into a single paragraph, without any line break or separator. "
    "Do NOT add any paragraph separator, blank line, or special character between paragraphs. Paragraph separation will be handled in postprocessing. "
    "NEVER add any text, sentence, word, phrase, punctuation, particle, ending, conjunction, symbol, or even a single character that does not exist in the PDF. "
    "If a sentence is cut off at a page break, do NOT complete or connect it unless the next page continues the same sentence or paragraph, in which case you MUST merge them as a single paragraph. Do NOT infer, imagine, supplement, auto-complete, connect, or grammatically correct any text. "
    "Output only the text that is actually visible in the PDF, exactly as it appears, preserving only paragraph boundaries. "
    "Do NOT output page numbers, page counts, footnote or annotation markers, repeated section headers, or any marginalia such as headers or footers. "
    "Ignore any symbols, numbers, or text that indicate page numbers, footnotes, endnotes, repeated subtitles, or appear in the page margins. "
    "IMPORTANT: Do NOT extract or output any text that appears inside images, tables, charts, diagrams, or figures, or captions. "
    "Do NOT extract or output any titles, captions, or explanatory text related to images, tables, charts, diagrams, or figures, even if they are outside the visual box. "
    "Do NOT extract or output any footnotes, endnotes, superscript numbers, reference marks, or annotation symbols, even if they appear in the main text or margins. "
    "For example, ignore any text like 'Figure 1', 'Note:', '※', '11', or any small numbers or symbols that indicate references or notes. "
    "Only extract the main body text. Ignore all visual elements and any text associated with them, including their titles, captions, and explanations. "
    "If you are unsure whether a text is part of a visual element or a note, do NOT extract it. "
    "Never output any text that breaks the natural flow of the main body paragraphs. "
    "For example, if there is a diagram, chart, or figure with labels or explanations inside a box or graphic, do NOT extract or output any of that text. "
    "If you see a diagram or chart, do not describe or transcribe it. "
    "You will be given a PDF file containing multiple book pages in order. For each page, extract only the main body text, in the order given. "
    "If a paragraph or sentence is split between two pages, merge them as a single paragraph, with no extra line break or separator. "
    "Do not merge or split content between unrelated paragraphs or sections. Output the extracted text for each page in the same order as the input pages, with no extra separators."
)

def get_last_saved_page(raw_text_dir):
    files = glob.glob(os.path.join(raw_text_dir, "*.txt"))
    if not files:
        return 0
    nums = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
    return max(nums)

def get_last_merged_page(merged_path):
    if not os.path.exists(merged_path):
        return 0
    with open(merged_path, "r", encoding="utf-8") as f:
        text = f.read()
    pages = text.split("\n\n*****\n\n")
    return len(pages)

def pipeline(book_root, book_name, dpi=300, window_size=1, resume=True, start_page: Optional[int]=None, end_page: Optional[int]=None, model: str = "gemini"):
    if model == "ollama-gemma":
        ensure_ollama_server()
        ensure_gemma_model()
    pdf_path = get_pdf_path(book_root, book_name)
    if not os.path.isfile(pdf_path):
        return
    img_dir = os.path.join(book_root, "images")
    ckpt_dir = os.path.join(book_root, "checkpoints")
    merged_dir = os.path.join(book_root, "merged")
    raw_text_dir = os.path.join(book_root, "raw_text")
    chunk_merged_dir = os.path.join(book_root, "chunk_merged")
    ensure_dir(img_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(merged_dir)
    ensure_dir(raw_text_dir)
    ensure_dir(chunk_merged_dir)
    from pdf_to_png import get_total_pages
    total_pages = get_total_pages(pdf_path)
    chunk_size = 30 if model == "gemini-2.5-pro" else 5
    chunk_indices = list(range(1, total_pages+1, chunk_size))
    chunk_merged_files = []
    handler = ModelHandlerFactory.get_handler(model)
    if model == "gemini-2.5-pro":
        pdf_chunks = split_pdf(pdf_path, pages_per_chunk=chunk_size)
        print(f"[DEBUG] PDF {len(pdf_chunks)}개로 분할 완료")
        for idx, chunk in enumerate(pdf_chunks, 1):
            print(f"  {idx}. {os.path.basename(chunk)}")
        # 전체 분할 PDF 경로와 인덱스 매핑
        chunk_idx_map = {os.path.abspath(chunk): idx for idx, chunk in enumerate(pdf_chunks, 1)}
        s = input("처리할 분할 PDF 시작 번호(엔터시 1): ").strip()
        e = input("처리할 분할 PDF 끝 번호(엔터시 전체): ").strip()
        start_idx = int(s) - 1 if s else 0
        end_idx = int(e) if e else len(pdf_chunks)
        selected_chunks = pdf_chunks[start_idx:end_idx]
        print(f"[DEBUG] {len(selected_chunks)}개 분할 PDF 처리 시작 ({start_idx+1}~{end_idx})")
        for chunk_path in selected_chunks:
            abs_chunk_path = os.path.abspath(chunk_path)
            global_idx = chunk_idx_map[abs_chunk_path]
            result = handler.extract_text([chunk_path], prompt=PDF_SYSTEM_PROMPT)
            print(f"[DEBUG] Gemini 2.5 Pro LLM 응답 수신 (part{global_idx})")
            text = result["text"]
            text = clean_text(text)
            text = gemini_paragraph_postprocess(text)
            # 분할 PDF별로 하나의 raw_text 파일로 저장 (고유 인덱스)
            raw_text_path = os.path.join(raw_text_dir, f"{book_name}_part{global_idx}.txt")
            with open(raw_text_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[DEBUG] RAW 텍스트 저장 완료: {os.path.basename(raw_text_path)}")
        # 병합
        merge_raw_text_pipeline(book_root, book_name)
        print(f"[완료] {book_name} 전체 파이프라인 종료.")
        return
    # 기존 flash/ollama-gemma 방식
    #print(f"\n[책 처리 시작] {book_name}")
    # 페이지 단위로 변환 및 추출
    from pdf_to_png import convert_pdf_to_images
    import time
    for page_num in range(1, total_pages+1):
        print(f"[DEBUG] {page_num}번 페이지 PNG 변환 시작")
        try:
            images = convert_pdf_to_images(
                pdf_path,
                output_dir=img_dir,  # 기존 이미지 폴더 사용
                dpi=dpi,
                first_page=page_num,
                last_page=page_num
            )
            if not images:
                print(f"[ERROR] {page_num}번 페이지 PNG 변환 실패")
                continue
            img_path = images[0]
            # 파일명을 0001.png, 0002.png ... 순서로 강제 리네이밍 (중복 방지)
            target_img_name = f"{page_num:04d}.png"
            target_img_path = os.path.join(img_dir, target_img_name)
            if os.path.abspath(img_path) != os.path.abspath(target_img_path):
                if not os.path.exists(target_img_path):
                    os.rename(img_path, target_img_path)
                else:
                    print(f"[WARN] {target_img_path} 이미 존재, 덮어쓰기 방지")
            else:
                target_img_path = img_path
            print(f"[DEBUG] {page_num}번 페이지 Gemini Flash 호출 시작: {os.path.basename(target_img_path)}")
            result = handler.extract_text([target_img_path])
            text = result.get("text", "").strip()
            # 후처리(postprocessing) 적용
            text = clean_text(text)
            text = gemini_paragraph_postprocess(text)
            raw_text_path = os.path.join(raw_text_dir, f"{page_num}.txt")
            with open(raw_text_path, "w", encoding="utf-8") as f:
                f.write(text)
            save_checkpoint(ckpt_dir, page_num-1, text)
            print(f"[DEBUG] {page_num}번 페이지 텍스트 저장 완료")
        except Exception as e:
            print(f"[ERROR] {page_num}번 페이지 처리 실패: {e}")
        time.sleep(30)  # 30초 대기

    # 최종 병합은 별도 함수로만 처리
    merge_raw_text_pipeline(book_root, book_name)

# 병합(merge)만 별도로 실행할 수 있는 함수 추가

def merge_raw_text(book_root, book_name):
    raw_text_dir = os.path.join(book_root, "raw_text")
    merged_dir = os.path.join(book_root, "merged")
    ensure_dir(merged_dir)
    files = sorted(glob.glob(os.path.join(raw_text_dir, "*.txt")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    #print(f"[DEBUG] 병합 대상 파일: {files}")
    merged = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            merged.append(f.read().strip())
    out_path = os.path.join(merged_dir, f"{book_name}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n*****\n\n".join(merged))
    print(f"[병합 완료] {out_path}")

def extract_part_num(filename):
    m = re.search(r'_part(\d+)$', os.path.splitext(filename)[0])
    return int(m.group(1)) if m else 0

def merge_raw_text_pipeline(book_root, book_name):
    raw_text_dir = os.path.join(book_root, "raw_text")
    merged_dir = os.path.join(book_root, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    merged_path = os.path.join(merged_dir, f"{book_name}.txt")
    files = sorted(
        glob.glob(os.path.join(raw_text_dir, "*.txt")),
        key=lambda x: extract_part_num(os.path.basename(x))
    )
    merged_text = ""
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            merged_text += f.read().strip() + "\n\n*****\n\n"
    with open(merged_path, "w", encoding="utf-8") as f:
        f.write(merged_text.strip())
    print(f"[병합 완료] {merged_path}")

def get_int_or_none(prompt):
    while True:
        s = input(prompt).strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            print("숫자를 입력하거나 엔터만 눌러 전체를 선택하세요.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF to Text 파이프라인 (Gemini/Ollama/Gemini 2.5 Pro 모델 선택 지원)")
    parser.add_argument("--book", type=str, help="책 폴더명(books/ 하위)")
    parser.add_argument("--start-page", type=int, default=None, help="시작 페이지 번호")
    parser.add_argument("--end-page", type=int, default=None, help="끝 페이지 번호")
    parser.add_argument("--model", type=str, choices=["gemini", "ollama-gemma", "gemini-2.5-pro"], default=None, help="텍스트 추출 모델 선택 (기본: gemini)")
    args = parser.parse_args()

    MODEL_CHOICES = [
        ("gemini", "Gemini 2.5 Flash"),
        ("ollama-gemma", "Gemma3:12b"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro")
    ]
    def select_model():
        print("\n[모델 선택] 사용할 텍스트 추출 모델을 선택하세요:")
        for idx, (mkey, mdesc) in enumerate(MODEL_CHOICES, 1):
            print(f"  {idx}. {mdesc} [{mkey}]")
        while True:
            sel = input("모델 번호 또는 이름 입력 (엔터시 1): ").strip()
            if not sel:
                return MODEL_CHOICES[0][0]
            if sel.isdigit():
                idx = int(sel) - 1
                if 0 <= idx < len(MODEL_CHOICES):
                    return MODEL_CHOICES[idx][0]
            for mkey, _ in MODEL_CHOICES:
                if sel.lower() == mkey:
                    return mkey
            print("잘못된 입력입니다. 다시 선택하세요.")

    books_root = "books"
    ensure_dir(books_root)
    books = list_books(books_root)
    # CLI 인자 우선 처리
    if args.book:
        if args.book not in books:
            print(f"{args.book} 폴더가 books/에 없습니다.")
            exit(1)
        model = args.model if args.model else select_model()
        pipeline(os.path.join(books_root, args.book), args.book, start_page=args.start_page, end_page=args.end_page, model=model)
    else:
        # 기존 input() 방식 fallback
        if not books:
            print(f"{books_root} 폴더에 책 폴더가 없습니다.")
            exit(1)
        print("\n[책 폴더 목록]")
        for idx, bname in enumerate(books, 1):
            print(f"{idx}. {bname}")
        print("0. 전체 자동 처리")
        sel = input("\n작업할 책 번호(또는 0 전체): ").strip()
        model = args.model if args.model else select_model()
        if sel == '0':
            for bname in books:
                pipeline(os.path.join(books_root, bname), bname, model=model)
        else:
            try:
                idx = int(sel) - 1
                if 0 <= idx < len(books):
                    bname = books[idx]
                    start_page = get_int_or_none("시작 페이지 번호(엔터시 전체): ")
                    end_page = get_int_or_none("끝 페이지 번호(엔터시 전체): ")
                    pipeline(os.path.join(books_root, bname), bname, start_page=start_page, end_page=end_page, model=model)
                else:
                    print("잘못된 번호입니다.")
            except Exception as e:
                print(f"[입력/실행 오류] {e}")
                import traceback; traceback.print_exc() 