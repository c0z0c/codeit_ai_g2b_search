import yaml
from pathlib import Path

class PromptLoader:
    """
    YAML 기반 프롬프트 템플릿 로더
    """

    def __init__(self, prompt_file: str = "src/llm/prompts/prompt_temp_v1.yaml"):
        # 현재 파일 경로 기준으로 project_root 자동 탐지
        # __file__ → .../src/llm/prompts/prompt_loader.py
        base_dir = Path(__file__).resolve()
        project_root = base_dir.parents[3]  # => .../codeit_ai_g2b_search/

        # prompt_file이 절대경로인지 체크
        prompt_path = Path(prompt_file)
        if not prompt_path.is_absolute():
            full_path = project_root / prompt_path
        else:
            full_path = prompt_path

        # 존재 확인
        if not full_path.exists():
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {full_path}")

        self.prompt_file = full_path
        self.templates = self._load_yaml()

    def _load_yaml(self):
        """YAML 파일 로드"""
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key: str):
        """특정 키의 프롬프트 템플릿 반환"""
        return self.templates.get(key)
