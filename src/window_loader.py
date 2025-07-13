import os
from typing import List, Callable, Any
import glob

def get_page_windows(image_paths: List[str], window_size: int = 3) -> List[List[str]]:
    """
    이미지 파일 리스트를 받아 N-1, N, N+1 윈도우(겹치는 window_size개)로 묶어 반환
    예: [1.png, 2.png, 3.png, 4.png] -> [[1,2,3], [2,3,4]]
    첫/마지막 페이지는 부족한 부분을 앞/뒤로 채움
    """
    n = len(image_paths)
    if n == 0:
        return []
    half = window_size // 2
    windows = []
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        # 경계 보정: 항상 window_size개가 되도록 앞/뒤 패딩
        win = image_paths[start:end]
        if len(win) < window_size:
            if start == 0:
                win = [image_paths[0]] * (window_size - len(win)) + win
            else:
                win = win + [image_paths[-1]] * (window_size - len(win))
        windows.append(win)
    return windows

def sequential_window_processing(
    image_paths: List[str],
    process_window_fn: Callable[[List[str], int], Any]
) -> List[Any]:
    """
    이미지 리스트를 윈도우 단위로 순차 처리
    process_window_fn(window_images, window_index)로 콜백 호출
    결과 리스트 반환
    """
    windows = get_page_windows(image_paths)
    results = []
    for idx, win in enumerate(windows):
        print(f"\n[윈도우 {idx+1}/{len(windows)}] {', '.join([os.path.basename(w) for w in win])}")
        result = process_window_fn(win, idx)
        results.append(result)
    return results

if __name__ == "__main__":
    # CLI: 이미지 폴더 입력받아 윈도우 순차 처리 예시
    img_dir = input("이미지 폴더 경로를 입력하세요 (예: page_txt/파일명): ").strip()
    all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not all_imgs:
        print("해당 폴더에 PNG 이미지가 없습니다.")
        exit(1)
    print(f"총 {len(all_imgs)}장 이미지 발견. 윈도우 단위로 순차 처리합니다.")

    def dummy_process(window_imgs, idx):
        # 실제로는 extract_text_from_image(window_imgs) 등으로 교체
        print(f"  처리 중: {', '.join([os.path.basename(w) for w in window_imgs])}")
        return f"윈도우{idx+1}: {', '.join([os.path.basename(w) for w in window_imgs])}"

    sequential_window_processing(all_imgs, dummy_process) 