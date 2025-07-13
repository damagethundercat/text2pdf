

import os
from pdf2image import convert_from_path, pdfinfo_from_path
from typing import List, Optional
from PIL.Image import Image

# 사용자가 제공한 poppler 경로
poppler_path = r"D:\Projects\AI\bin\poppler-24.08.0\Library\bin"

def get_total_pages(pdf_path: str) -> int:
    """PDF 파일의 총 페이지 수를 반환합니다."""
    try:
        info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
        return info["Pages"]
    except Exception as e:
        print(f"Error getting page count: {e}")
        return 0

def load_nested_pages(pdf_path: str, page_number: int, dpi: int = 300) -> Optional[List[Image]]:
    """
    주어진 PDF 파일에서 N, N-1, N+1 페이지를 이미지로 로드합니다.

    :param pdf_path: PDF 파일 경로
    :param page_number: 대상 페이지 번호 (1부터 시작)
    :param dpi: 이미지 해상도 (dots per inch)
    :return: PIL Image 객체의 리스트, 또는 오류 발생 시 None
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return None

    total_pages = get_total_pages(pdf_path)
    if total_pages == 0 or page_number > total_pages or page_number < 1:
        print(f"Error: Invalid page number {page_number}. Total pages: {total_pages}.")
        return None

    # 페이지 번호는 1-based, pdf2image는 1-based
    first_page = max(1, page_number - 1)
    last_page = min(total_pages, page_number + 1)

    print(f"Loading pages from {first_page} to {last_page}...")

    try:
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=first_page,
            last_page=last_page,
            fmt="png",
            poppler_path=poppler_path
        )
        
        # 반환된 이미지 리스트와 요청한 페이지 범위가 일치하는지 확인
        # 예: 1페이지를 요청하면 first_page=1, last_page=2 -> 이미지 2장 반환
        # page_number가 1일 때, first_page=1, last_page=2. images[0]은 1페이지, images[1]은 2페이지.
        # page_number가 중간일 때, first_page=N-1, last_page=N+1. images[0]은 N-1, images[1]은 N, images[2]는 N+1 페이지.
        
        return images
    except Exception as e:
        print(f"An error occurred while converting PDF to images: {e}")
        return None

# --- 여기서부터 코드를 실행합니다 ---
if __name__ == "__main__":
    # 테스트할 PDF 파일과 페이지 번호 지정
    pdf_file = os.path.join("pdf", "네번째불연속_300dpi.pdf")
    target_page = 5

    print(f"Total pages in '{pdf_file}': {get_total_pages(pdf_file)}")
    
    # 네스팅된 페이지 로드
    nested_images = load_nested_pages(pdf_file, target_page)

    if nested_images:
        print(f"Successfully loaded {len(nested_images)} images for page {target_page}.")
        
        # 결과 이미지 저장 폴더 생성
        output_dir = os.path.join("pdf", "extract", "image_test")
        os.makedirs(output_dir, exist_ok=True)

        # 로드된 이미지들을 파일로 저장하여 확인
        for i, img in enumerate(nested_images):
            # 페이지 번호는 first_page부터 시작
            current_page_num = max(1, target_page - 1) + i
            img_path = os.path.join(output_dir, f"page_{current_page_num}.png")
            img.save(img_path, "PNG")
            print(f"Saved image for page {current_page_num} to {img_path}")

    else:
        print("Failed to load nested pages.")


