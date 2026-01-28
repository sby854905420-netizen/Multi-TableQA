"""
    This file contains the PromptBuilder class, which is used to build prompts for the model to generate questions from.
    It mainly contains the following parts:
    1. __init__: Initialize PromptBuilder
    2. add_text: Add text to prompt
    3. add_question: Add question
    4. clear: Clear Prompt
    5. get_prompt: Get Prompt
    6. __str__: Returns the full prompt when printing
    
    7. default_format: Default output formatting instructions for code output
    8. add_format: Add output formatting instructions to the prompt
    9. default_task_description: Default task description
    10. add_task_description: Add task description to the prompt
    11. default_table_desciption: Default table description
    12. add_table_description: Add table description to the prompt
"""


class PromptBuilder:


    def __init__(self, base_content: str = ""):
        """
        Initialize PromptBuilder
        :param base_content: initial content (empty by default)
        """
        self.content = base_content

    def add_text(self, text: str, newline: bool = True):
        """
        Add text to prompt
        :param text: the text to add
        :param newline: whether to add a newline at the end
        """
        if newline:
            self.content += "\n" + text
        else:
            self.content += text
        return self  

    def add_question(self, question: str):

        """
        Add question
        :param question: The question to be added
        """
        return self.add_text("Based on the information provided above, answer the following question:\n" + question)

    def clear(self):
        """
        Clear Prompt
        """
        self.content = ""

    def get_prompt(self) -> str:
        """
        Get Prompt
        """
        return self.content

    def __str__(self):
        """
        Returns the full prompt when printing
        """
        return self.get_prompt()
    
    def default_format(self) -> str:
        
        return """Print all key-value pairs, ensuring no truncation.
        **Important Formatting Rules:**

        - Ensure all values are fully printed, with no omissions or truncation (`...`).
        - Use the exact format as shown below:

        {
            "CASEID": 6042,
            "CRASHYEAR": 2017,
            "PSU": 32,
            "CASENO": 1,
            "CASENUMBER": "1-32-2017-001-09",
            "CATEGORY": 9,
            "CRASHMONTH": 1,
            "DAYOFWEEK": 1,
            "CRASHTIME": "08:42",
            "CONFIG": 5,
            "EVENTS": 1,
            "VEHICLES": 1,
            "CAIS": 2,
            "CISS": 9,
            "CINJURED": 1,
            "CINJSEV": 0,
            "CTREAT": 3,
            "ALCINV": 2,
            "DRGINV": 2,
            "MANCOLL": 0,
            "SUMMARY": "V1 was negotiating a curve and traveling on the connection ramp of two major highways. Driver lost directional control of the vehicle. V1 started to rotate counter-clockwise and departed the roadway to the left. The front of V1 impacted a concrete traffic barrier.",
            "CASEWGT": 439.9888081,
            "PSUSTRAT": 7,
            "VERSION": 2
        }
        """
    def add_format(self, format_text: str = None):
        """
        Add formatting instructions to the prompt.
        :param format_text: The formatting instructions to be added (default: use self.default_format()).
        """
        if format_text is None:
            format_text = self.default_format()  

        return self.add_text(format_text)
    
    def default_task_description(self) -> str:
        
        return """
        Please help me write a code to access xixi_first_test.xlsx and the file path is 'data/xixi_first_test.xlsx'. 
        """
        
    def add_task_description(self, task_description: str = None):
        
        if task_description is None:
            task_description = self.default_task_description()  

        return self.add_text(task_description)
    
    def default_table_desciption(self) -> str:
        
        return """ The desciprtion of the table is as follows:
        Each row contains information of a crash. The table contains various attributes related to crash cases, 
        including CASEID, CRASHYEAR, PSU, CASENO, CASENUMBER, CATEGORY, 
        CRASHMONTH, DAYOFWEEK, CRASHTIME, CONFIG, EVENTS, VEHICLES, CAIS, 
        CISS, CINJURED, CINJSEV, CTREAT, ALCINV, DRGINV, MANCOLL, SUMMARY, 
        CASEWGT, PSUSTRAT, and VERSION.
        """

    def add_table_description(self, table_description: str = None):
        """
        Add table description to the prompt.
        :param table_description: The table description to be added (default: use self.default_table_description()).
        """
        if table_description is None:
            table_description = self.default_table_desciption()  

        return self.add_text(table_description)


if __name__ == "__main__":

    captured_output = "Vehicle 1 was traveling northbound on the roadway and struck a pedestrian at an intersection."

    Question = "What was the direction of travel of Vehicle 1?"

    builder = PromptBuilder(base_content=captured_output)
    builder.add_question(Question)

    print(builder.get_prompt())

