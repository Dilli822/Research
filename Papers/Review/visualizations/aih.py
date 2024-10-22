import matplotlib.pyplot as plt

# Function to create a timeline plot
def create_timeline(milestones, color, title):
    """
    Create a timeline plot for a set of AI milestones.

    Parameters:
    - milestones: Dictionary with years as keys and event descriptions as values.
    - color: Color of the line and markers.
    - title: Title of the plot.
    """
    # Extract years and event names
    years = list(milestones.keys())
    events = list(milestones.values())

    # Create the timeline plot
    plt.figure(figsize=(12, 8))  # Adjust figure size for better layout
    plt.plot(years, list(range(len(years))), marker='o', color=color, linestyle='-', linewidth=2)

    # Annotating each event
    for i, event in enumerate(events):
        plt.text(years[i], i, event, fontsize=12, verticalalignment='center', ha='right')

    # Adding labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Event Order', fontsize=14)

    # Set y-ticks as the event numbers and x-ticks as the years
    plt.yticks(list(range(len(years))), [])  # Y-axis shows event positions
    plt.xticks(years, rotation=45)  # Rotate x-axis labels for better readability

    # Show the grid only for x-axis
    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(False)

    # Adjust layout to avoid clipping
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.85)

    plt.show()

# Data for progressive milestones, AI winters, and modern vs classical AI
progressive_events = {
    1956: "Dartmouth Conference (Birth of AI)",
    1966: "ELIZA Chatbot",
    1997: "Deep Blue Defeats Kasparov",
    2006: "Hinton's Deep Learning Revival",
    2011: "IBM Watson Wins Jeopardy!",
    2012: "AlexNet Wins ImageNet",
    2020: "GPT-3 Released",
    2023: "ChatGPT Reaches 100M Users"
}

downfall_events = {
    1973: "First AI Winter",
    1987: "Second AI Winter"
}

modern_ai_events = {
    2011: "IBM Watson Wins Jeopardy",
    2012: "AlexNet Wins ImageNet",
    2020: "GPT-3 Released",
    2023: "ChatGPT Reaches 100M Users"
}

classical_ai_events = {
    1956: "Dartmouth Conference",
    1966: "ELIZA (First Chatbot)",
    1980: "First National AAAI Conference",
    1997: "Deep Blue Defeats Kasparov"
}

# ---- Plot 1: Progressive Milestones ----
create_timeline(progressive_events, color='green', title='Progressive Milestones in AI')

# ---- Plot 2: Downfall Events (AI Winters) ----
create_timeline(downfall_events, color='red', title='Downfall Events in AI (AI Winters)')

# ---- Plot 3: Modern vs Classical AI ----
# Plot Modern AI
create_timeline(modern_ai_events, color='blue', title='Modern AI Milestones')

# Plot Classical AI
create_timeline(classical_ai_events, color='orange', title='Classical AI Milestones')


# https://chatgpt.com/share/6715d681-b648-8007-ab0d-f88286b6c88b