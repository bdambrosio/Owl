def determine_thread(user_input, active_threads, conversation_history, llm):
    # Prepare the prompt for the LLM
    prompt = f"User Input: {user_input}\n\n"
    
    # Add active threads to the prompt
    prompt += "Active Threads:\n"
    for thread_name, thread_description in active_threads.items():
        prompt += f"- {thread_name}: {thread_description}\n"
    
    # Add conversation history to the prompt
    prompt += "\nConversation History:\n"
    for interaction in conversation_history:
        prompt += f"User: {interaction['user']}\n"
        prompt += f"Assistant: {interaction['assistant']}\n\n"
    
    # Add instructions for the LLM
    prompt += "Instructions:\n"
    prompt += "1. Analyze the user input and determine the most relevant thread from the active threads.\n"
    prompt += "2. If the user input introduces a new topic or thread, suggest a new thread name.\n"
    prompt += "3. Provide the thread name or the new thread suggestion as the output.\n"
    
    # Ask the LLM to determine the thread
    llm_response = llm.ask(prompt)
    
    # Process the LLM's response
    llm_response = llm_response.strip()
    
    if llm_response in active_threads:
        # User input belongs to an existing thread
        return llm_response
    else:
        # User input introduces a new thread
        new_thread_name = llm_response
        return new_thread_name
