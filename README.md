# PDF to Text (AI 기반 대용량 PDF 텍스트 추출기)

## 소개
이 프로젝트는 대용량 PDF 파일(수백~수천 페이지)을 AI 기반 멀티모델(Gemini 2.5 Flash, Gemini 2.5 Pro, Gemma 3:12B 등)로 빠르고 정확하게 텍스트로 변환하는 파이프라인입니다.

- **멀티모델 지원:** Google Gemini, Ollama-Gemma 등 다양한 모델 선택 가능
- **대용량 처리:** 800페이지 이상의 PDF도 체크포인트/병합/후처리로 안정적으로 처리
- **CLI 기반 사용:** 명령줄에서 간편하게 실행, 다양한 옵션 제공
- **API 비용/성능/한도 관리:** 실제 테스트 기반의 비용/성능/한도 우회 전략 내장

---

## 설치 방법
1. Python 3.9 이상 설치
2. 필수 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```
3. (Windows) [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) 설치 후 환경변수 POPPLER_PATH 설정
4. `.env` 파일에 Google API Key 등 환경변수 설정
   ```env
   GOOGLE_API_KEY=your_google_api_key
   ```

---

## 사용법

### 1. 명령줄 인자 사용 (비대화식)
```bash
python src/pdf2text_pipeline.py --book <책폴더명> [--model <모델명>] [--start-page N] [--end-page M]
```
- `--book`: books/ 하위 폴더명 (예: singularity)
- `--model`: gemini (2.5 Flash), gemini-2.5-pro, ollama-gemma 중 선택 (기본: gemini)
- `--start-page`, `--end-page`: 처리할 페이지 범위 지정(선택)

### 2. 대화형 메뉴 사용 (인자 없이 실행)
```bash
python src/pdf2text_pipeline.py
```
- 책 목록이 번호로 출력 → 번호 입력
- 모델 선택 메뉴(번호/이름/엔터) → 선택
- 시작/끝 페이지 입력(엔터시 전체)
- 0 입력 시 전체 자동 처리

### 3. 예시
```bash
python src/pdf2text_pipeline.py --book singularity --model gemini
# 또는
python src/pdf2text_pipeline.py
# → 대화형 메뉴로 책/모델/페이지 선택
```

---

## 지원 모델
- **Gemini 2.5 Flash** (Google, 빠르고 저렴, 이미지→텍스트)
- **Gemini 2.5 Pro** (Google, PDF 직접 입력, 고품질)
- **Gemma 3:12B** (Ollama, 로컬, 무료, 인식률 낮음)

> 실제 사용시 API 비용/한도/성능을 반드시 확인하세요.

---

## 폴더 구조 및 준비 방법

- **books/** 폴더 내부에 원하는 책 폴더(예: singularity, 의식이라는꿈 등)를 직접 생성하세요.
- 각 책 폴더 안에 변환할 PDF 파일(예: singularity.pdf 등)을 넣어야 합니다.
- 예시:
```
books/
  ├─ singularity/
  │    └─ singularity.pdf
  └─ 의식이라는꿈/
       └─ 의식이라는꿈.pdf
```

> 책 폴더명과 PDF 파일명은 자유롭게 지정할 수 있습니다. 단, CLI 실행 시 --book 옵션에 폴더명을 정확히 입력해야 합니다.

---

## 환경 변수
- `.env` 파일에 아래 항목 필수
  - `GOOGLE_API_KEY` : Gemini API 키

---

## 의존성
- pdf2image
- google-generativeai
- python-dotenv
- tqdm
- requests
- pypdf
- Pillow

---

## 라이선스
MIT License

---

## 참고/기여
- Pull Request, Issue 환영합니다!
- 실제 대용량 PDF, 다양한 모델 테스트 결과/팁 공유해주시면 감사하겠습니다. 
