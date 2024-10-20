import matplotlib.pyplot as plt

def create_timeline(milestones):
    """
    Create a timeline visualization for a list of events and years.
    
    Parameters:
    milestones (list of tuples): List where each tuple contains an event and the corresponding year.
    """
    # Extracting data for visualization
    years = [milestone[1] for milestone in milestones]
    events = [milestone[0] for milestone in milestones]

    # Create the timeline visualization
    plt.figure(figsize=(12, 8))  # Adjusting figure size for better responsiveness
    plt.plot(years, list(range(len(years))), marker='o', color='green')

    # Annotating each event
    for i, event in enumerate(events):
        plt.text(years[i], i, event, fontsize=14, verticalalignment='center', ha='right')

    # Adding labels and title
    plt.title('Timeline of AI Milestones', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Timeline', fontsize=14)
    plt.yticks(list(range(len(years))), years)
    # plt.yticks(range(len(years)), events)
    plt.grid(True)
    # Enabling only the y-axis grid
    # plt.gca().yaxis.grid(True)  # Show grid for y-axis
    # plt.gca().xaxis.grid(False)  # Hide grid for x-axis
    # Adjust layout for better appearance
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjusting right margin to avoid clipping of text

    plt.show()

# Example usage with AI timeline data:
milestones = [
    ("Turing’s Contributions", 1930),
    ("Cybernetics and Early Computing", 1940),
    ("Birth of AI", 1956),
    ("Rise of Neural Networks", 1960),
    ("Expert Systems and AI Winters", 1970),
    ("Revival of AI", 1980),
    ("Modern AI and Machine Learning", 1990),
    ("Deep Learning Revolution", 2010),
    ("Current AI Developments and Pursuit of AGI", 2020),
]

# Call the function to create the timeline
create_timeline(milestones)
