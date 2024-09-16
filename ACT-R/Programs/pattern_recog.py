from python_actr import *

class PatternAgent(ACTR):
    sequence = Buffer()
    recognized_pattern = Buffer()

    def __init__(self):
        super().__init__()
        self.sequence.set('A B B A A B B')

    def recognize_pattern(self):
        seq = self.sequence.get()  # Retrieve the contents of the buffer
        # Convert the buffer contents to a string for easier pattern matching
        seq_string = seq[0] if seq else ''
        if 'B B A' in seq_string:
            self.recognized_pattern.set('Pattern found: B B A')
        else:
            self.recognized_pattern.set('Pattern not found')

    def end(self):
        result = self.recognized_pattern.get()  # Retrieve the result from the buffer
        print(result)

    def run(self):
        self.recognize_pattern()
        self.end()

# Create and run the agent
agent = PatternAgent()
agent.run()
