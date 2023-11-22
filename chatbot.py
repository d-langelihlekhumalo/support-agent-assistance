from flask import Flask, render_template_string, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI    
from dotenv import load_dotenv
import json
import os

app = Flask(__name__)
app.static_folder = 'static'

# Load dotenv and get access to the API KEY
load_dotenv()

# Constants to hold file names
CLICKATELL_RAW_DATA = "clickatell_data.txt"
CLICKATELL_VECTORSTORE = "clickatell_data.db"
AGENT_FEEDBACK = "agent_feedback.txt"

if not os.path.exists(CLICKATELL_RAW_DATA) and not os.path.exists(CLICKATELL_VECTORSTORE):
    print("File for embeddings and datastore not found")
    os._exit(0)

def read_clickatell_data_from_file():
    extracted_file_data = ""
    with open(CLICKATELL_RAW_DATA, 'r', encoding='utf-8') as data_in_file:
        extracted_file_data = data_in_file.read()
    return extracted_file_data

def read_agent_feedback_data_from_file():
    extracted_file_data = ""
    with open(AGENT_FEEDBACK, 'r', encoding='utf-8') as data_in_file:
        extracted_file_data = data_in_file.read()
    return extracted_file_data

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, 
    chunk_overlap=200, 
    separators = ["\n\n", "\n", " ", ""])

embeddings = OpenAIEmbeddings()

doc_search = None
if not os.path.exists(CLICKATELL_VECTORSTORE):
    # Get raw data from clickatell file 
    raw_text = read_clickatell_data_from_file()
    texts = text_splitter.split_text(raw_text)

    doc_search = FAISS.from_texts(texts, embeddings)     
    doc_search = FAISS.save_local(CLICKATELL_VECTORSTORE)
else:
    doc_search = FAISS.load_local(CLICKATELL_VECTORSTORE, embeddings)

if os.path.exists(AGENT_FEEDBACK):
    raw_text = read_agent_feedback_data_from_file()

if raw_text.strip():
    # Embed the agent feedback and store in vector
    texts = text_splitter.split_text(raw_text)
    doc_search.add_texts(texts)


chat_history_UI = []

template = """

You are a helpful assistant for a company called Clickatell, use formal language.
You are refered to as Clickybot. 
All of your answers should be unique, new and custom. You are not allowed to repeat yourself
You adapt your answers through corrective feedback which is provided in the context.
Use the corrective feedback to improve and enrich your responses while still ensuring that all answers remain unique 

Use the following pieces of context to answer the users question. 
You must always consult the followng rules before answering:
- Context provided could include previously asked questions that are similar. Consider the answers provided when crafting your unique answers.
- Answers to previously asked questions should be seen as more relevant in the context
- Consider the chat history when replying for added context. 
- If the question doesnt relate to the provided documents or context, say "I'm sorry but the information requested is out of scope."
- If the user asks a question that is not within Clickatell services, say "I'm sorry but the information requested is out of scope." 
- You are not allowed to repeat yourself or give the same answer.

    
    {context}

Chat history:
{chat_history}
Follow up question from user: {user_question}
ClickyBot:
"""



@app.route("/")
def index():
    chat_history_str = "\n".join(chat_history_UI)
    return render_template_string(
        """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Chat with Customer Service</title>
        <link
        rel="stylesheet"
        href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"/>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.4/jquery.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>

        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f2f2f2;
                margin: 0;
                padding-top: 50px;
                text-align: center;
            }

            h1 {
                margin-bottom: 20px;
                text-align: center;
            }

            #chat-container {
                background-color: #fff;
                max-height: 450px;
                height: 450px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 10px;
                text-align: left;
                margin-bottom: 20px;
            }

            .message {
                margin-bottom: 10px;
                display: flex;
                align-items: flex-start;
            }

            .user-message {
                display: flex;
                justify-content: flex-end;
            }

            .bot-message {
                justify-content: flex-start;
            }
                                    
            .feedback-message {
                justify-content: flex-start;
            }

            .message-bubble {
                max-width: 70%;
                padding: 10px;
                border-radius: 10px;
                margin: 5px;
            }

            .icon-bubble {   
                margin-left: 5px; 
            }
            
            .user-bubble {
                background-color: #007bff;
                border-radius: 15px 15px 0 15px;
                color: #fff;
            }

            .bot-bubble {
                background-color: #e5e5ea;
                border-radius: 15px 15px 15px 0;
                color: #333;
            }
                                    
            .feedback-bubble {
                background-color: #e5e5ea;
                border-radius: 15px 15px 15px 0;
                color: #FFFF00;
            }

            .user-icon {
                width: 30px;
                height: 30px;
                margin-right: 5px;
            }

            .bot-icon {
                width: 40px;
                height: 30px;
                margin-right: 5px;
            }
                                    
            #inputText {
                width: 80%;
                padding: 10px;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 10px;
            }

            button {
                padding: 10px 20px;
                background-color: #007bff;
                color: #fff;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }

            button:hover {
                background-color: #0056b3;
            }

        </style>

    </head>

    <body>
        <div class="container-fluid" id="container">
            <h1>Chat with Customer Service</h1>
            <div class="container-fluid" id="chat-container">
                              
                <!-- user-message -->
                {% if user_input %}
                <div class="message user-message" id="userInput">
                    <div class="message-bubble user-bubble">
                        {{ user_input }}
                    </div>                    
                </div>
                {% endif %} 

                <!-- bot_response -->
                {% if bot_answer %}
                <div class="message bot-message" id="botResponse">  
                    <div class="message-bubble bot-bubble">
                    {{ bot_answer }}
                    </div>
                </div>
                {% endif %}
                   
        </div>
        <input type="text" id="inputText" placeholder="Type your message here">
        <button onclick="processInput()">Send</button>
    </div>

    <!-- The Modal -->
    <div class="modal" id="myModal"  data-backdrop="static" data-keyboard="false" tabindex="-1" aria-labelledby="staticBackdropLabel" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">

            <!-- Modal Header -->
            <div class="modal-header">
                <h4 class="modal-title" style="text-align: center; margin: 0 auto;">Review Bots Response</h4>
                <!--<button type="button" class="close" data-dismiss="modal">&times;</button>-->
            </div>

            <!-- Modal body -->
            <div class="modal-body">
                <div class="alert alert-success">
                    <p id="users_question"></p>
                </div>

                <form >
                    <div class="form-group">
                        <label for="bots_response">Bots Response:</label>
                        <textarea class="form-control" id="bots_response" name="bots_response" rows="4" cols="50"></textarea>
                        <small id="emailHelp" class="form-text text-muted">Please review the bots response and modify it if necessary.</small>
                    </div>

                    <div class="row" style="padding-top: 10px;">
                        <div class="container">
                            <div class="btn-group">
                                <button type="button" class="btn btn-danger" id="btnClearResponse" onclick="clearResponse()">Clear Response</button>
                                <button type="button" class="btn btn-warning" id="btnUndoCleareance" onclick="undoCleareance()">Undo Cleareance</button>
                                <button type="button" class="btn btn-success" id="btnSubmitResponse" onclick="saveResponsesToChatHistory()">Submit Response</button>
                            </div>
                        </div> 
                    </div>
                </form>
            </div>
            </div>
        </div>
    </div>
                              
    <script>
        function handleKeyPress(event) {
            if (event.keyCode === 13) {
                processInput();
            }
        }

        var keepBotsResponse = "";
        var actualBotsResponse = "";
        var currentUserQuestion = "";
        
        function triggerReviewSection(user_question, bots_response) {
            var myModal = document.getElementById('myModal');
            document.getElementById('users_question').innerHTML = user_question;
            document.getElementById('bots_response').value = bots_response;
            keepBotsResponse = bots_response;
            actualBotsResponse = bots_response;
            document.getElementById('btnUndoCleareance').disable = true;
            $(myModal).modal('show');
        }

        function clearResponse() {
            keepBotsResponse = document.getElementById('bots_response').value;
            document.getElementById('bots_response').value = '';
            document.getElementById('btnClearResponse').disable = true;
            document.getElementById('btnUndoCleareance').disable = false;
        }

        function undoCleareance() {
            document.getElementById('bots_response').value = keepBotsResponse;
            document.getElementById('btnClearResponse').disable = false;
            document.getElementById('btnUndoCleareance').disable = true;
        }

        function isEmpty(val){
            return (val === undefined || val == null || val.length <= 0) ? true : false;
        }       

        function saveResponsesToChatHistory() {
            bots_response = document.getElementById('bots_response').value;
            helpMessage = document.getElementById('emailHelp');
            if (isEmpty(bots_response)) {
                helpMessage.classList.remove('text-muted');
                helpMessage.classList.add('text-danger');
                helpMessage.innerHTML = 'Please ensure you have provided an appropriate response';
                return null;
            }

            helpMessage.classList.remove('text-danger');
            helpMessage.classList.add('text-muted');
            helpMessage.innerHTML = 'Please review the bots response and modify it if necessary.';
            
            // Append the user's message to the chat history
            //appendToChatHistory(currentUserQuestion, "user-message");

            // Update the botResponse div
            appendToChatHistory(bots_response, "bot-message");
                            
            // Clear the input field
            document.getElementById("inputText").value = "";
            var myModal = document.getElementById('myModal');
            $(myModal).modal('hide');
            saveInteraction(currentUserQuestion, actualBotsResponse, bots_response);
        }

        function saveInteraction(user_question, bots_response, correction) {
            if (bots_response == correction) {
                correction = "None";
                return;
            }
            data = {
                "user_question": user_question,
                "answer": bots_response,
                "correction": correction
            }

            fetch('/user_question', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            })
            .then(response => response.json())
            .then(data => {
                // Handle the response from the server if needed
                document.getElementById("inputText").value = "";
            })
            .catch(error => console.error('Error:', error));
        }

        // Add event listener to the input field
        document.getElementById("inputText").addEventListener("keydown", handleKeyPress);
        
        async function processInput() {
            var inputText = document.getElementById("inputText").value;

            // Append the user's message to the chat history
            appendToChatHistory(inputText, "user-message");

            // Clear the input field
            document.getElementById("inputText").value = "";

            try {
                const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ user_input: inputText }),
                });
                
                if (!response.ok) {
                    throw new Error('Unable to fetch bot response')
                }
                
                const data = await response.json();

                currentUserQuestion = inputText;
                triggerReviewSection(inputText, data.bot_response)
            } catch(err) {
                console.error('Error:', err);
            }   

        }

        function appendToChatHistory(message, messageType) {
            var chatContainer = document.getElementById("chat-container");

            // Create a new message element
            var messageDiv = document.createElement("div");
            messageDiv.classList.add("message", messageType);
            
            // Create an icon bubble element
            var iconBubble = document.createElement("div");
            iconBubble.classList.add("icon-bubble");

            // Styling icons
            var iconImg = document.createElement("img");
            if (messageType === "user-message") {
                iconImg.src = "/static/images/user_icon.png";
                iconImg.alt = "User Icon";
                iconImg.classList.add("user-icon");
            } else if (messageType === "bot-message") {
                iconImg.src = "/static/images/bot_icon.png";
                iconImg.alt = "Bot Icon";
                iconImg.classList.add("bot-icon");
            }
            iconBubble.appendChild(iconImg);

            // Create a message bubble element
            var messageBubble = document.createElement("div");
            messageBubble.classList.add("message-bubble", messageType === "user-message" ? "user-bubble" : "bot-bubble");
            messageBubble.innerText = message;

            // Append the icon bubble and message bubble to the message element
            if (messageType === "user-message") {
                messageDiv.appendChild(messageBubble);
                messageDiv.appendChild(iconBubble);
            } else {
                messageDiv.appendChild(iconBubble);
                messageDiv.appendChild(messageBubble);
            }
            // Append the message element to the chat history container
            chatContainer.appendChild(messageDiv);
        }
        </script>
    </body>
</html>
""",
        chat_history_UI=chat_history_str,
        user_input="",
        bot_answer="",
    )


prompt = PromptTemplate(
    input_variables=["chat_history", "user_question", "context"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_question")

chain = load_qa_chain(
    ChatOpenAI(temperature= 1.0),
    chain_type="stuff",
    memory=memory,
    prompt=prompt,
    verbose=True,
)


def write_feedback_response_to_file(user_question, ai_response, correction=None):
    data = {
        "user_question": user_question,
        "answer": ai_response
    }
    if correction is not None:
        data["correction"] = correction

    with open("agent_feedback.txt", "a+", encoding="utf-8") as file:
        file.seek(0)
        is_empty = not bool(file.read())
        file.seek(0, 2)
        
        if not is_empty:
            file.write(",\n")
        json.dump(data, file, indent=4)

    file.close()


@app.route('/user_question', methods=['POST'])
def user_question():
    try:
        user_question = request.get_json()
       
        if user_question['correction'] == "None":
            write_feedback_response_to_file(user_question['user_question'], user_question['answer'])
        else:
            write_feedback_response_to_file(user_question['user_question'], user_question['answer'], user_question['correction'])
            # memory.buffer.exchange[-1] = (user_question['user_question'], user_question['correction'])
            # memory.buffer[-1] = (user_question['user_question'], user_question['correction'])
            correction = {"user_question":{user_question['user_question']},"answer":{user_question['answer']}, "correction": {user_question['correction']}}
            texts=text_splitter.split_text(str (correction))
            doc_search.add_texts(texts)
        return jsonify({"message": "Data received successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def chat():
    try:
        user_question = request.json["user_input"]

        docs = doc_search.similarity_search(user_question)
        response = chain({"input_documents": docs, "user_question": user_question})
        bot_answer = response["output_text"]

        chat_history_UI.append({user_question})
        chat_history_UI.append({bot_answer})

        return (
            jsonify({"user_question": user_question, "bot_response": bot_answer}),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
