from llm_loader import LLM

class PromptGenerator:
    def __init__(self, analysis_model="llama3.2:1b", question: str = ""):
        """
        PromptGenerator initialization
        :param analysis_model: the name of the small model used to analyze the problem
        :param question: questions to be analyzed
        """
        self.analysis_model = analysis_model
        self.question = question
        self.llm = LLM(model_name=self.analysis_model)

    def analyze_question(self, question: str):
        """
        Use a small model to analyze the problem and generate prompts of different lengths
        :param question: the question entered by the user
        :return: long prompt, medium prompt, short prompt
        """
        analysis_prompt = (
            f"Please analyze the following question based on provided information and generate prompts of different granularity:\n{question}\n"
             "Generate a complete and detailed prompt, a medium-length prompt, and a concise prompt."
        )
        
        response = self.llm.query(analysis_prompt)

        return response

    
    
    def add_basic_knowledge(self, prompt: str, basic_knowledge: str) -> str:    
        """
        Add basic knowledge to prompt
        :param prompt: original prompt
        :param basic_knowledge: basic knowledge to be added
        :return: prompt after adding basic knowledge
        """
        if basic_knowledge is None or basic_knowledge.strip() == "":
            raise ValueError("Basic knowledge cannot be None or empty.")
        return f"{basic_knowledge}\n{prompt}"
    

    def add_text(self, prompt: str, text: str) -> str:
        """
        Add text to prompt
        :param prompt: original prompt
        :param text: text to be added
        :return: prompt after adding text
        """
        return f"{prompt}\n{text}"




