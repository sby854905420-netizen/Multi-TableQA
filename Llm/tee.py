import io

class Tee:
    
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = io.StringIO()

    def write(self, text):
        self.original_stdout.write(text)  
        self.buffer.write(text)  

    def flush(self):
        self.original_stdout.flush()