# Blog-Generation-Using-LLAMA-2

This repository contains code for a Blog Generation Platform using the Llama 2 models. This README file provides a comprehensive explanation of the code, the technologies used, and the concepts behind it.

## Table of Contents
1. [Introduction](#introduction)
2. [Technologies Used](#technologies-used)
3. [Code Explanation](#code-explanation)
4. [Setup Instructions](#setup-instructions)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

The Llama 2 release introduces a family of pretrained and fine-tuned LLMs, ranging from 7B to 70B parameters. These models are trained on 40% more tokens than their predecessors, have a context length of up to 4k tokens, and feature grouped-query attention for fast inference of the 70B model. This project utilizes these models to create an end-to-end blog generation platform.

## Technologies Used

- **Python**: The primary programming language used.
- **Transformers**: A library by Hugging Face used for NLP tasks.
- **Llama 2**: The family of language models used for generating content.
- **Streamlit**: A framework for creating web applications.
- **Pandas**: Used for data manipulation and analysis.
- **NumPy**: Used for numerical operations.

## Code Explanation

Below is the complete code for the Blog Generation Platform along with explanations for each section.

```python
# Importing necessary libraries
import streamlit as st  # Streamlit is used to create the web interface.
from transformers import LlamaTokenizer, LlamaForCausalLM  # Hugging Face's Transformers library for working with Llama 2 models.
import torch  # PyTorch library, required for model operations.

# Function to load the Llama model and tokenizer
@st.cache(allow_output_mutation=True)  # Caches the function to avoid reloading the model on each interaction.
def load_model(model_name):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)  # Loads the tokenizer for the specified Llama model.
    model = LlamaForCausalLM.from_pretrained(model_name)  # Loads the language model for causal language modeling.
    return tokenizer, model

# Setting up the Streamlit interface
st.title("Blog Generation Platform")  # Sets the title of the web app.
st.markdown("## Generate a blog post using Llama 2")  # Adds a markdown section.

# Input for the blog topic
prompt = st.text_input("Enter the topic for your blog")  # Creates a text input field for entering the blog topic.

# Load the Llama model and tokenizer
tokenizer, model = load_model("Llama2-model")  # Calls the load_model function to load the tokenizer and model.

# Generate blog content based on the input topic
if st.button("Generate Blog"):  # Adds a button to generate the blog post.
    inputs = tokenizer(prompt, return_tensors="pt")  # Tokenizes the input prompt.
    outputs = model.generate(inputs.input_ids, max_length=500)  # Generates text based on the tokenized input with a maximum length of 500 tokens.
    blog_post = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decodes the generated tokens into a readable string, skipping special tokens.
    st.write(blog_post)  # Displays the generated blog post.


