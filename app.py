import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import pickle
import os
import random 

# Initialize stemmer
stemmer = LancasterStemmer()


def load_data():
    with open('data.pickle2', 'rb') as f:
        return pickle.load(f)

# Load the model
def load_model():
    input_size = len(words)
    hidden_size = 8
    output_size = len(labels)

    class ChatBotModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(ChatBotModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return self.softmax(x)

    model = ChatBotModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device)  
    model.eval()
    return model


if not os.path.exists('data.pickle2') or not os.path.exists('model.pth'):
    st.write("Model or data files not found. Please check the setup.")
    st.stop()


words, labels, _, _ = load_data()  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model()


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag, dtype=np.float32)


with open('./indian_restaurant.json') as file:
    data = json.load(file)


st.title("Restaurant Chatbot")

if 'history' not in st.session_state:
    st.session_state.history = []
if 'welcomed' not in st.session_state:
    st.session_state.welcomed = False

if not st.session_state.welcomed:
    welcome_prompt = st.text_input("Say 'hello' to start the conversation:", key="welcome_prompt")
    if welcome_prompt.lower() == 'hello':
        welcome_response = "Welcome! I am here to assist you with restaurant-related questions. How can I help you today?"
        st.session_state.history.append({'question': welcome_prompt, 'answer': welcome_response})
        st.write(f"**User😍:** {welcome_prompt}")
        st.write(f"**Bot😎:** {welcome_response}")
        st.session_state.welcomed = True
else:
    
    for entry in st.session_state.history:
        st.write(f"**User😍:** {entry['question']}")
        st.write(f"**Bot😎:** {entry['answer']}")

    # Prompt for new user input
    prompt = st.text_input("Enter your prompt:", key="conversation_prompt")

    if prompt:
        
        bag = bag_of_words(prompt, words)
        bag_tensor = torch.tensor(bag, dtype=torch.float32).unsqueeze(0).to(device)

        
        with torch.no_grad():
            model.eval()
            results = model(bag_tensor)
        results_index = torch.argmax(results).item()
        tag = labels[results_index]

      
        responses = None
        for tg in data['indian_restaurant']:
            if tg['tag'] == tag:
                responses = tg['responses']
                break
        
        if responses:
            answer = random.choice(responses)
        else:
            answer = "Sorry, I didn't understand that."

        st.session_state.history.append({'question': prompt, 'answer': answer})
        st.write(f"**User😍:** {prompt}")
        st.write(f"**Bot😎:** {answer}")
