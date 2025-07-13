import os
import re
from typing import List

def clean_text(text: str) -> str:
    """
    텍스트 정제: 불필요한 공백, 연속 띄어쓰기, 특수문자 정리 등
    """
    # 줄 끝 공백 제거, 연속 공백/탭 정리
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    # 특수문자(제어문자 등) 제거(필요시 조정)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # 여러 줄바꿈 2개로 통일(문단 구분)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_paragraphs(text: str, delimiter: str = '\n\n') -> List[str]:
    """
    빈 줄(2개 이상) 기준으로 문단 분리
    """
    paras = [p.strip() for p in text.split(delimiter) if p.strip()]
    return paras

def process_txt_file(input_path: str, output_path: str, delimiter: str = '\n\n'):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    cleaned = clean_text(raw)
    paras = split_paragraphs(cleaned, delimiter)
    # 문단 구분자: \n\n*****\n\n
    result = ('\n\n*****\n\n').join(paras)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"[처리 완료] {os.path.basename(input_path)} → {os.path.basename(output_path)}")

def batch_process_txt_folder(input_dir: str, output_dir: str, delimiter: str = '\n\n'):
    os.makedirs(output_dir, exist_ok=True)
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
    for fname in txt_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        process_txt_file(in_path, out_path, delimiter)

def cli_select_and_process(input_dir: str, output_dir: str, delimiter: str = '\n\n'):
    os.makedirs(output_dir, exist_ok=True)
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
    if not txt_files:
        print("해당 폴더에 txt 파일이 없습니다.")
        return
    print("\n[TXT 파일 목록]")
    for idx, fname in enumerate(txt_files, 1):
        print(f"{idx}. {fname}")
    while True:
        try:
            sel = int(input("\n변환할 TXT 번호를 입력하세요: "))
            if 1 <= sel <= len(txt_files):
                break
            else:
                print("잘못된 번호입니다. 다시 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    selected = txt_files[sel-1]
    in_path = os.path.join(input_dir, selected)
    base, ext = os.path.splitext(selected)
    out_path = os.path.join(output_dir, f"{base}_pp{ext}")
    process_txt_file(in_path, out_path, delimiter)

def gemini_paragraph_postprocess(text: str) -> str:
    """
    Gemini가 문단마다 \n만 넣는 경우, 모든 \n을 \n\n*****\n\n으로 치환
    """
    return text.replace('\n', '\n\n*****\n\n').strip()

if __name__ == "__main__":
    print("1. 폴더 내 전체 일괄 처리\n2. 파일 선택 후 단일 변환")
    mode = input("모드를 선택하세요(1/2): ").strip()
    in_dir = input("입력 txt 폴더 경로를 입력하세요: ").strip()
    out_dir = input("출력 폴더 경로를 입력하세요: ").strip()
    if mode == '1':
        batch_process_txt_folder(in_dir, out_dir)
    else:
        cli_select_and_process(in_dir, out_dir) 