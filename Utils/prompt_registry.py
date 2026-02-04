import yaml
from jinja2 import Template
from pathlib import Path

class PromptRegistry:
    def __init__(self, prompt_dir: str):
        self.prompt_dir = Path(prompt_dir)
        self._cache = {}

    def load(self, prompt_id: str):
        if prompt_id in self._cache:
            return self._cache[prompt_id]

        for path in self.prompt_dir.glob("*.yaml"):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data.get("id") == prompt_id:
                    self._cache[prompt_id] = data
                    return data

        raise ValueError(f"Prompt id not found: {prompt_id}")

    def render(self, prompt_id:str, schema_description:str, questoin:str, evidence:str):
        prompt = self.load(prompt_id)
        system_msg = prompt["system"]

        user_tpl = Template(prompt["user_template"])
        example_tpl = Template(prompt["examples"])

        user_msg = user_tpl.render({
            "schema_rows":schema_description,
            "question":questoin,
            "tips":evidence,
        })
        example_msg = example_tpl.render({
            "example_schema_rows":schema_description
        })


        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg + "\n\n" + example_msg},
        ]

    @staticmethod
    def hash_prompt(messages):
        text = "\n".join(m["content"] for m in messages)
        # return hashlib.sha256(text.encode("utf-8")).hexdigest()
        return text
    
    