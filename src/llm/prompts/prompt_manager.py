class PromptManager:
    def __init__(self, prompt_dir: str = "src/llm/prompts", env: str = "prod"):
        """
        프로젝트 루트 기준으로 prompt 디렉터리 자동 탐지
        """
        # Notebook 환경에서는 __file__이 없으므로 안전하게 부모 탐색으로 프로젝트 루트 결정
        if "__file__" in globals():
            base_dir = Path(__file__).resolve()
        else:
            base_dir = Path.cwd()

        # 부모 폴더를 위로 탐색하여 프로젝트 루트(=src 폴더 포함)를 찾음
        project_root = base_dir
        for p in [base_dir] + list(base_dir.parents):
            if (p / "src").exists() or (p / ".git").exists():
                project_root = p
                break

        full_dir = project_root / prompt_dir
        env_dir = full_dir / env if (full_dir / env).exists() else full_dir

        if not env_dir.exists():
            raise FileNotFoundError(f"Prompt 디렉토리가 없습니다: {env_dir}")

        self.prompt_dir = env_dir

    def list_templates(self) -> list[str]:
        """템플릿 파일 이름 목록"""
        return [f.stem for f in self.prompt_dir.glob("*.yaml")]

    def load_template(self, template_id: str) -> Dict[str, Any]:
        """지정한 템플릿 로드"""
        template_file = self.prompt_dir / f"{template_id}.yaml"
        if not template_file.exists():
            raise FileNotFoundError(f"템플릿 파일이 없습니다: {template_file}")
        with open(template_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data

    def render_template(self, template_id: str, values: Dict[str, str]) -> str:
        """템플릿에 값 채워넣기"""
        tpl = self.load_template(template_id)
        template_str = tpl.get("template", "")
        try:
            rendered = template_str.format(**values)
        except KeyError as e:
            missing = e.args[0]
            raise ValueError(f"템플릿에 필요한 값이 없습니다: {missing}")
        return rendered