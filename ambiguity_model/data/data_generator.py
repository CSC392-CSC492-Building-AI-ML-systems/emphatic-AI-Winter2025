# Generate distinct ambiguous and unambiguous prompts
import random
import pandas as pd


ambiguous_prompts_distinct = [
    "Can you explain what I mean?",
    "What’s the best way to handle it?",
    "How does that impact things?",
    "What’s the usual outcome?",
    "Tell me what you think about this.",
    "Can this be done differently?",
    "Is there a right way to proceed?",
    "What’s the expected result?",
    "What would you suggest?",
    "Does this seem correct to you?",
    "Is there more to consider?",
    "What should I focus on?",
    "Would this work in most cases?",
    "How would one typically approach this?",
    "What do you mean by that?",
    "Is this relevant here?",
    "What’s the usual way to go about it?",
    "How do people generally feel about it?",
    "What’s the standard response to this?",
    "Could you clarify further?",
    "Is this always the case?",
    "What’s the next step?",
    "Would you say this is correct?",
    "How does one know for sure?",
    "Is there a specific method for this?",
    "What if things change?",
    "Would this apply universally?",
    "Is this common?",
    "What do others think about this?",
    "How important is it?",
    "Is this always true?",
    "Would you say this is significant?",
    "Can you shed some light on this?",
    "What’s the general approach to this?",
    "Does this require special attention?",
    "How should I interpret this?",
    "What would be a reasonable assumption?",
    "How does this compare to other cases?",
    "What are the key considerations here?",
    "How can I tell if this is right?",
    "What’s the general sentiment about it?",
    "Should I be concerned about this?",
    "Is there a common solution?",
    "How does one typically respond?",
    "What do you mean exactly?",
    "Is this something to worry about?",
    "Would this be appropriate?",
    "How do people usually react to this?",
    "Does this align with expectations?",
    "Would you say this is typical?",
    "How can I be sure?",
    "What’s the general opinion on this?",
    "Is there a better way?",
    "Does this make a difference?",
    "Would this work in any scenario?",
    "How should I think about this?",
    "What’s your view on this?",
    "Does this make a strong case?",
    "Is this a reasonable question?",
    "What should I watch out for?",
    "How do experts handle this?",
    "Is there a rule for this?",
    "Would this be a mistake?",
    "Is this something I should pursue?",
    "How do I know I’m on the right track?",
    "What’s your best guess?",
    "Does this need further clarification?",
    "Would this typically be the case?",
    "Should I explore other options?",
    "How would you rate this?",
    "Is this the right call?",
    "What’s your impression of this?",
    "Could this go either way?",
    "Would this be enough?",
    "What’s the safest bet?",
    "Is this a well-known issue?",
    "Does this suggest something deeper?",
    "How can I validate this?",
    "Is this a common mistake?",
    "How do you see this playing out?",
    "Would this require adjustments?",
    "Should I reconsider?",
    "What’s the underlying assumption?",
    "Is there a pattern here?",
    "Is this a gray area?",
    "Would this be fair?",
    "Does this hold up under scrutiny?",
    "How might this change over time?",
    "Would you call this a best practice?",
    "How do I test this idea?",
    "What’s a critical factor here?",
    "Does this conflict with anything else?",
    "Is this an accurate representation?",
    "What’s a typical challenge with this?",
    "Could this lead to unexpected results?",
    "Is this the main issue?",
    "What’s a common oversight here?",
    "Would this apply broadly?",
    "How do you typically verify this?",
    "Could this be misinterpreted?",
    "What’s the key takeaway?",
    "Would this create any problems?",
    "Is there an exception to this?",
    "Should I be looking for alternatives?",
    "How flexible is this rule?",
    "Is this the best comparison?",
    "Does this stand up to scrutiny?",
    "Would this cause confusion?",
    "What’s a good analogy for this?",
    "Could this indicate a larger trend?",
    "Would you categorize this differently?",
    "Is this universally accepted?",    
]

unambiguous_prompts_distinct = [
    f"Explain {topic} in detail."
    for topic in [
        "quantum mechanics", "the history of World War II", "the difference between classical and operant conditioning",
        "how photosynthesis works", "the importance of cybersecurity", "how to bake a chocolate cake",
        "the role of mitochondria in a cell", "how blockchain technology ensures security", "the Pythagorean theorem",
        "Newton's laws of motion", "the process of DNA replication", "how electric cars work",
        "the causes and effects of climate change", "the significance of the Renaissance",
        "how machine learning algorithms make predictions", "the role of the immune system", "the major causes of inflation",
        "the theory of evolution", "how the human brain processes language", "the impact of social media on communication",
        "how vaccines work", "the structure of the solar system", "the economic principles of supply and demand",
        "how a combustion engine functions", "the physics behind rainbows", "how deep learning models function",
        "the importance of financial literacy", "how search engines rank web pages",
        "the impact of the Industrial Revolution", "the main differences between reptiles and amphibians",
        "how fiber optics transmit data", "the process of volcanic eruptions", "the concept of entropy in thermodynamics",
        "how the stock market operates", "how the Mars rover collects data", "how to set up a Linux server",
        "the principles of aerodynamics", "how antibiotics work", "how the periodic table is organized",
        "the effects of caffeine on the human body", "the cultural impact of Shakespeare", "how earthquakes are measured",
        "the principles of quantum entanglement", "how solar panels generate electricity",
        "how genes determine traits", "the benefits of a ketogenic diet", "how an internal combustion engine works",
        "how black holes are formed", "the fundamental concepts of game theory", "the process of metamorphosis in insects",
        "the role of neurotransmitters in the brain"
    ]
]

# Create dataset ensuring distinct entries
data_distinct = [[prompt, 0] for prompt in ambiguous_prompts_distinct] + \
                [[prompt, 1] for prompt in unambiguous_prompts_distinct]

# Shuffle dataset
random.shuffle(data_distinct)

# Convert to DataFrame
df_distinct = pd.DataFrame(data_distinct, columns=["text", "label"])

# Save the dataset to a CSV file
file_path_distinct = "ambiguous_prompts_dataset_distinct.csv"
df_distinct.to_csv(file_path_distinct, index=False)

# Display the dataset to the user
# tools.display_dataframe_to_user(name="Distinct Ambiguous Prompts Dataset", dataframe=df_distinct)

# file_path_distinct
