from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_chroma import Chroma
import os

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Initialize LLM and Embeddings
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-04-17', google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Initialize vectorstore function
def vectorstore(collection_name, directory):
    return Chroma(
        collection_name=collection_name,
        persist_directory=directory,
        embedding_function=embeddings
    )

# Load all vector stores
vs1 = vectorstore('insurance1.vdb', 'insurance1.db')
vs2 = vectorstore('insurance2.vdb', 'insurance2.db')
vs3 = vectorstore('insurance3.vdb', 'insurance3.db')
vs4 = vectorstore('insurance4.vdb', 'insurance4.db')
vs5 = vectorstore('insurance5.vdb', 'insurance5.db')

# Retrieve relevant docs from each vector store
def retrieve(question):
    retrievers = [
        vs1.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
        vs2.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
        vs3.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
        vs4.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
        vs5.as_retriever(search_type='similarity', search_kwargs={'k': 5})
    ]
    return [retriever.invoke(question) for retriever in retrievers]

# Format chat history
def format_chat_history(history):
    if not history:
        return "No previous conversation."
    return "\n".join([
        f"User: {item['user_query']}\nAssistant: {item['ai_response']}"
        for item in history
    ])

# Prompt Template
prompt_template = """
You are a friendly and intelligent AI Insurance Assistant.

Your job is to help users choose the most appropriate insurance policy based on the **five company-provided policy documents** and the **user's current question**. Use the user’s **past chat history** (if available) to understand preferences or previous questions.

---

### 🏢 COMPANIES:
- Company 1: SURYAJYOTI
- Company 2: LIC NEPAL
- Company 3: SANIMA
- Company 4: MET LIFE
- Company 5: SUNLIFE

### 📄 DOCUMENT CONTEXTS:
{context1}
{context2}
{context3}
{context4}
{context5}

### 💬 USER QUESTION:
{question}

### 🔁 CHAT HISTORY:
{chat_history}

---

### INSTRUCTIONS:

You are an intelligent and friendly insurance advisor chatbot. Your goal is to recommend the most suitable insurance plan to the user based on their age, location, and specific needs (health, critical illness, term life, affordability, etc.).

Follow this structure for every response:

1. 🧠 Understand the User  
   If their intent or needs are unclear, ask relevant clarifying questions.

2. 📊 Present a Brief Comparison Table  
   Display a simple comparison table of 2–3 suitable insurance plans:
   
   | Plan Name | Company | Type | Age Range | Unique Benefit |
   |-----------|---------|------|-----------|----------------|
   | Example A | LIC NEPAL | Health + Critical Illness | 18–64 | Covers 18 major diseases |
   | Example B | MET LIFE | Critical Illness | 18–54 | Lump sum payout on diagnosis |

3. ✅ Recommend the Best Plan  
   Briefly explain why it fits the user's needs best.

4. 📌 Briefly Mention Other Good Options  
  Point out there also other 1-2 plans if only available and also why they can prefer or chosse this as well.

5.Engage and personalize:


   End with a friendly follow-up such as:


   “Would you like to explore this plan in more detail?”


   “Would you prefer a plan with lower premiums or one with broader critical illness coverage?”


   “Let me know what matters most to you — cost, illness coverage, or something else?”

6.Context-aware follow-ups:
   If the user responds positively (e.g., “Yes,” “Tell me more,” or shares preferences), look at the chat history and context to give a deeper and more tailored explanation of the previously recommended insurance plan.
   Use the user’s past responses to personalize the explanation further, such as referencing their age, disease concern, or prior budget-related messages.

Maintain a warm, helpful, and conversational tone throughout, making the user feel guided rather than sold to. Avoid technical jargon unless necessary, and explain terms simply when used.


#     ## FORMATTING GUIDELINES:


#     - Use proper **Markdown formatting**:
#     - **Headings** (`###`, `##`) for organizing sections.
#     - **Bold** and *italic* for emphasis.
#     - **Bullet points** and **tables** for easy comparison.
#     - Keep text visually clean, skimmable, and helpful.
#     - For code or parameter values, use triple backticks where applicable.



"""

# Initialize the template
prompt = PromptTemplate.from_template(prompt_template)


# Modify the generate_response function to use passed history
def generate_response(contexts, question, history):
    formatted_history = format_chat_history(history)
    formatted_prompt = prompt.format(
        context1=contexts[0],
        context2=contexts[1],
        context3=contexts[2],
        context4=contexts[3],
        context5=contexts[4],
        question=question,
        chat_history=formatted_history
    )
    return llm.invoke(formatted_prompt).content

# Modify the main chatbot function
def get_chatbot_response(user_message, history=None):
    if history is None:
        history = []

    # Step 1: Retrieve from vectorstores
    docs = retrieve(user_message)
    contexts = [''.join(doc.page_content + '\n' for doc in doc_list) for doc_list in docs]

    # Step 2: Generate response with history
    response = generate_response(contexts, user_message, history)

    return response
