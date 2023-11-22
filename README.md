# RaD-GP-C23-P-I6

## Project Name

AI Support Agent Recommendation | with RLFH

## Project Overview

This project is about developing an intelligent chatbot trained on company data.
The chatbot is enhanced with RLHF to help it improve it's recommendations over time.

### 1. Chatbot Integration | Implementation

The basic chatbot is setup using the langchain framework. It can accept one or more text files and
will embed them using OpenAIEmbeddings, thereafter store the embeddings into a FAISS vectorstore.
The chatbot is built using python and streamlit for easy interaction. Conversational memory is
implemented to help the chatbot remember conversation history, templating also implemented to provide
more context to the chatbot. It will store the user questions and the responses from the llm with the ratings into a text file that is seperate from the company data file and pass it to the vector database.

### 2. Support Agent Recommendation

The main purpose of the chatbot is to provide recommendations to support agents. Users (customers)
send questions, queries through to the support agent, the chatbot will pick it up, process it
and provide a recommendation (answer) to the support agent the best it can. Depending on the quality
of the recommendation, the support agent will rate the recommendation and use or discard it accordingly.

## 3. RLHF Technique Implementation

We implemented RLHF using the Help Along the Way method, where we have our Chat bot answer questions, thereafter we allow support agents to rate the response (positive or negative), if the response generated is negative, we allow human intervention where support agents can provide the appropriate response thereafter, we save the question, feedback, and appropriate response for later reference to further improve the chatbots recommendations. Through this implementation, the chatbot can learn from its prior mistakes and improve as it encounters similar questions.

### Challenges of RLHF:

RLHF models are subjected to certain limitations, such as:

1. Training Bias: RLHF models might be prone to algorithmic bias. Complex political or philosophical queries can have several answers, but the model will stick to its default training answers, resulting in bias.
2. Subjectivity of Human Feedback: Human feedback is subjective and varies from trainer to trainer. Therefore, RLHF models are prone to inconsistencies and human error. Creating training guidelines and working with experts could be a possible solution.
3. Scalability: Since the process relies on human feedback, training large-scale and more complex models requires extensive resources and time.
4. Inaccurate Answers: The accuracy and quality of answers depend on human annotations. It’s tricky for an AI chatbot to understand user intent. As a result, the generated text might be incorrect unless you input the exact wording used during training.
   There is room for improvement in addressing the limitations of RLHF models, which can lead to greater effectiveness and reliability in their application.

## How to Install and Run

- Please clone the repository:
  - Open a _terminal_ or _command prompt_, type the following to clone the project

```sh
    git clone https://gitlab1a.prod.eu-west-1.aws.clickatell.com/innovation/rad-gp-c23-p-i6-t2.git
```

- Navigate to the projects directory, in the terminal ty

```sh
    cd rad-gp-c23-p-i6-t2
```

- Once inside of the _project directory_:
  - On Windows, follow these steps:

```sh
    venv\bin\activate
    pip install -r requirements.txt
    python chatbot.py
```

- On Unix, follow these steps:
- Replace _python3_ with installed python version

```sh
    source venv\bin\activate
    pip install -r requirements.txt
    python3 chatbot.py
```

- If the above methods don't work _(Windows or Unix)_:
- Ensure you are inside the _project folder_
- On Windows, omit the python3 and use Python
- On Unix, use the appropriate installed python version

```sh
    pip install langchain python-dotenv faiss-cpu openai colorama
    python3 chatbot.py
```