import os
import platform
from pdf2image import convert_from_path, pdfinfo_from_path
from typing import List, Optional
from PIL.Image import Image
from dotenv import load_dotenv
load_dotenv()

# poppler 경로 자동 탐색 함수

def get_poppler_path():
    # 1. 환경변수 우선
    poppler_path = os.environ.get("POPLER_PATH")
    if poppler_path and os.path.exists(poppler_path):
        return poppler_path
    # 2. OS별 기본 경로 자동 탐색
    system = platform.system()
    if system == "Windows":
        possible_paths = [
            r"C:\\Program Files\\poppler-24.08.0\\Library\\bin",
            r"C:\\Program Files\\poppler-23.11.0\\Library\\bin",
            r"C:\\Program Files\\poppler-0.68.0\\bin",
        ]
    elif system == "Darwin":  # macOS
        possible_paths = [
            "/opt/homebrew/bin",  # Apple Silicon
            "/usr/local/bin",    # Intel Mac
        ]
    else:  # Linux
        possible_paths = ["/usr/bin", "/usr/local/bin"]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    # 3. 못 찾으면 안내
    raise RuntimeError("Poppler 경로를 찾을 수 없습니다. POPPLER_PATH 환경변수를 직접 지정해 주세요.")

POPLER_PATH = get_poppler_path()


def get_total_pages(pdf_path: str) -> int:
    """PDF 파일의 총 페이지 수 반환"""
    try:
        info = pdfinfo_from_path(pdf_path, poppler_path=POPLER_PATH)
        return info["Pages"]
    except Exception as e:
        print(f"[ERROR] 페이지 수 확인 실패: {e}")
        return 0


def convert_pdf_to_images(
    pdf_path: str,
    output_dir: str = "page_txt/png_output",
    dpi: int = 300,
    resize_max_width: Optional[int] = None,
    sequential_naming: bool = True,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None
) -> List[str]:
    """
    PDF 전체 또는 일부 페이지를 PNG로 변환하여 저장. 파일명은 0001.png, 0002.png 등.
    :param pdf_path: PDF 파일 경로
    :param output_dir: 저장 폴더
    :param dpi: 해상도
    :param resize_max_width: 최대 가로 크기(초과 시 리사이즈)
    :param sequential_naming: True면 0001.png 등 순차 네이밍
    :param first_page: 변환 시작 페이지(1부터, None이면 전체)
    :param last_page: 변환 끝 페이지(None이면 전체)
    :return: 저장된 파일 경로 리스트
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일이 존재하지 않습니다: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    total_pages = get_total_pages(pdf_path)
    if total_pages == 0:
        raise ValueError("PDF 페이지 수를 확인할 수 없습니다.")

    try:
        kwargs = dict(
            pdf_path=pdf_path,
            dpi=dpi,
            fmt="png",
            poppler_path=POPLER_PATH
        )
        if first_page is not None:
            kwargs["first_page"] = first_page
        if last_page is not None:
            kwargs["last_page"] = last_page
        images = convert_from_path(**kwargs)
        saved_files = []
        for idx, img in enumerate(images, 1):
            # 리사이즈 옵션 적용
            if resize_max_width and img.width > resize_max_width:
                ratio = resize_max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((resize_max_width, new_height))
            # 파일명 생성
            if sequential_naming:
                filename = f"{idx:04d}.png"
            else:
                filename = f"page_{idx}.png"
            out_path = os.path.join(output_dir, filename)
            img.save(out_path, "PNG")
            saved_files.append(out_path)
        return saved_files
    except Exception as e:
        print(f"[ERROR] PDF → 이미지 변환 실패: {e}")
        raise


def convert_nested_pages(
    pdf_path: str,
    page_number: int,
    output_dir: str = "page_txt/nested_output",
    dpi: int = 300
) -> List[str]:
    """
    N-1, N, N+1 페이지를 PNG로 저장
    :param pdf_path: PDF 파일 경로
    :param page_number: 기준 페이지(1부터)
    :param output_dir: 저장 폴더
    :param dpi: 해상도
    :return: 저장된 파일 경로 리스트
    """
    total_pages = get_total_pages(pdf_path)
    if total_pages == 0 or page_number > total_pages or page_number < 1:
        raise ValueError(f"잘못된 페이지 번호: {page_number} (총 {total_pages} 페이지)")
    first_page = max(1, page_number - 1)
    last_page = min(total_pages, page_number + 1)
    os.makedirs(output_dir, exist_ok=True)
    try:
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=first_page,
            last_page=last_page,
            fmt="png",
            poppler_path=POPLER_PATH
        )
        saved_files = []
        for i, img in enumerate(images):
            current_page = first_page + i
            filename = f"nested_{current_page:04d}.png"
            out_path = os.path.join(output_dir, filename)
            img.save(out_path, "PNG")
            saved_files.append(out_path)
        return saved_files
    except Exception as e:
        print(f"[ERROR] 중첩 페이지 변환 실패: {e}")
        raise

if __name__ == "__main__":
    # CLI: sample_pdfs 폴더 내 PDF 파일 리스트업 및 선택
    pdf_dir = "sample_pdfs"
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"{pdf_dir} 폴더에 PDF 파일이 없습니다.")
        exit(1)
    print("\n[PDF 파일 목록]")
    for idx, fname in enumerate(pdf_files, 1):
        print(f"{idx}. {fname}")
    while True:
        try:
            sel = int(input("\n변환할 PDF 번호를 입력하세요: "))
            if 1 <= sel <= len(pdf_files):
                break
            else:
                print("잘못된 번호입니다. 다시 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    selected_pdf = pdf_files[sel-1]
    pdf_path = os.path.join(pdf_dir, selected_pdf)
    pdf_name_wo_ext = os.path.splitext(selected_pdf)[0]
    output_dir = os.path.join("page_txt", pdf_name_wo_ext)
    print(f"\n선택된 PDF: {selected_pdf}")
    print(f"이미지 저장 폴더: {output_dir}")
    # 특정 페이지 번호 입력받기
    total_pages = get_total_pages(pdf_path)
    print(f"총 페이지 수: {total_pages}")
    while True:
        try:
            page_num = int(input(f"\n변환할 기준 페이지 번호를 입력하세요 (1~{total_pages}): "))
            if 1 <= page_num <= total_pages:
                break
            else:
                print("잘못된 페이지 번호입니다. 다시 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    try:
        out_files = convert_nested_pages(pdf_path, page_num, output_dir=output_dir)
        print(f"{len(out_files)}개 이미지 저장 완료! (페이지: {max(1, page_num-1)}~{min(total_pages, page_num+1)})")
    except Exception as e:
        print(f"[실행 오류] {e}") 