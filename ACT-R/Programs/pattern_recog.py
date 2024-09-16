# import random

# class Memory:
#     def __init__(self):
#         self.chunks = {}

#     def add_chunk(self, name, content):
#         self.chunks[name] = content

#     def retrieve(self, name):
#         return self.chunks.get(name, None)

# class Production:
#     def __init__(self, name, condition, action):
#         self.name = name
#         self.condition = condition
#         self.action = action

# class ACTR:
#     def __init__(self):
#         self.memory = Memory()
#         self.productions = []
#         self.goal = None
#         self.visual_buffer = None

#     def add_production(self, production):
#         self.productions.append(production)

#     def set_goal(self, goal):
#         self.goal = goal

#     def set_visual_input(self, visual_input):
#         self.visual_buffer = visual_input

#     def run(self):
#         while self.goal:
#             for production in self.productions:
#                 if production.condition(self):
#                     print(f"Firing production: {production.name}")
#                     production.action(self)
#                     break

# # Define productions
# def see_shape(model):
#     return model.visual_buffer is not None and model.goal == "identify-shape"

# def identify_shape(model):
#     shape = model.visual_buffer
#     model.memory.add_chunk("current-shape", shape)
#     model.goal = "respond-to-shape"
#     print(f"Shape identified: {shape}")

# def respond_to_shape(model):
#     return model.goal == "respond-to-shape" and model.memory.retrieve("current-shape") is not None

# def give_response(model):
#     shape = model.memory.retrieve("current-shape")
#     response = "press_button" if shape == "circle" else "do_nothing"
#     print(f"Responding to {shape}: {response}")
#     model.goal = None

# # Create and set up the model
# model = ACTR()
# model.add_production(Production("see-shape", see_shape, identify_shape))
# model.add_production(Production("respond-to-shape", respond_to_shape, give_response))

# # Run the model
# def run_trial(shape):
#     print(f"\nTrial with shape: {shape}")
#     model.set_goal("identify-shape")
#     model.set_visual_input(shape)
#     model.run()

# # Run multiple trials
# for _ in range(3):
#     run_trial(random.choice(["circle", "square"]))


from python_actr import *

class PatternAgent(ACTR):
    sequence = Buffer()
    recognized_pattern = Buffer()

    def init(self):
        self.sequence.set('A B B A A B B')
        self.goal.set('recognize')

    def recognize_pattern(self, sequence='A B B A A B B', pattern='B B A'):
        if pattern in sequence:
            self.recognized_pattern.set('Pattern found: ' + pattern)
        else:
            self.recognized_pattern.set('Pattern not found')
        self.goal.set('end')

    def end_recognition(self):
        print(self.recognized_pattern.chunk['name'])
        self.stop()

agent = PatternAgent()
agent.goal.set('recognize')
agent.run()