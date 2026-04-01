# LangGraph + Agentic AI: Customer Support Agent with Multiple Tools + Streamlit UI
# -------------------------------------------------------------------------------
# Tools: FAQ retrieval, Order Status Checker (via CSV), Sentiment Analyzer (LLM), Email Escalation, General Chatbot

import pandas as pd
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from email.mime.text import MIMEText
from dotenv import load_dotenv
import smtplib
import os
import re

# Load env variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# LLM Setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0.4)

# ---------------------------
# Load FAQ PDF into Vector DB
# ---------------------------
def load_faq_vector_db():
    loader = PyPDFLoader("Customer_Support_FAQs.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, GoogleGenerativeAIEmbeddings(model="embedding-001", google_api_key=google_api_key))
    return vectorstore

faq_vector_db = load_faq_vector_db()

# ---------------------------
# Tools Definitions
# ---------------------------
def check_order_status(order_id: str) -> str:
    df = pd.read_csv("order_data.csv")
    order = df[df['order_id'] == order_id]
    if order.empty:
        return "Order ID not found. Please check and try again."
    row = order.iloc[0]
    return f"Order Status for {order_id}: {row['status']}, Estimated Delivery: {row['eta']}"

def analyze_sentiment(text: str) -> str:
    prompt = PromptTemplate.from_template("""
    Analyze the sentiment of the following customer message:
    "{message}"
    Is it Positive, Neutral, or Negative?
    Just reply with one word: Positive, Neutral, or Negative.
    """)
    result = llm.invoke(prompt.format(message=text))
    return str(getattr(result, "content", result)).strip().lower()

def escalate_to_support(email: str, issue: str) -> str:
    try:
        msg = MIMEText(f"Customer Issue:\n{issue}")
        msg['Subject'] = "Support Escalation Request"
        msg['From'] = "noreply@supportbot.com"
        msg['To'] = email
        smtp = smtplib.SMTP('localhost')
        smtp.sendmail(msg['From'], [msg['To']], msg.as_string())
        smtp.quit()
        return f"Escalation email sent to {email}"
    except Exception as e:
        return f"Failed to send email: {e}"

def get_faq_response(query: str) -> str:
    return faq_vector_db.similarity_search(query, k=1)[0].page_content

def general_chat_response(message: str) -> str:
    prompt = f"""You are a helpful and friendly customer support assistant. 
    Greet the customer and ask the customer of any help that you can provide. 
    Answer briefly and clearly. Do not answer anything outside of the customer queries related to the orders. 
    Tell them to stick to the orders info if they ask you something out of domain information.:\n{message}"""
    result = llm.invoke(prompt.format(message=message))
    return str(getattr(result, "content", result)).strip().lower()

# ---------------------------
# LangChain Agent with Tools
# ---------------------------
tools = [
    Tool(
        name="CheckOrderStatus",
        func=lambda x: check_order_status(x),
        description="Use this to check the status of an order only if user provides order ID."
    ),
    Tool(
        name="FAQTool",
        func=lambda x: get_faq_response(x),
        description="Use this to answer frequently asked questions like returns, warranty, payment, etc."
    ),
    Tool(
        name="GeneralChat",
        func=lambda x: general_chat_response(x),
        description="Always use this if the user is making a general greeting, asking a question, or just chatting casually."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-zero-shot-react-description",
    verbose=False,
    max_iterations=5
)

# ---------------------------
# Streamlit Chat Interface
# ---------------------------
st.set_page_config(page_title="Support Agent AI", page_icon="🤖", layout="centered")
st.title("🤖 Smart Customer Support Bot")

st.markdown("""<style>body { background-color: #f0f8ff; }</style>""", unsafe_allow_html=True)

user_input = st.text_input("Talk to your support assistant:")

if user_input:
    st.markdown(f"**You:** {user_input}")
    sentiment = analyze_sentiment(user_input)
    st.markdown(f"<span style='color:orange'>Sentiment: {sentiment.title()}</span>", unsafe_allow_html=True)

    response = agent.invoke(user_input)
    st.write(response)

    if sentiment == "negative":
        user_email = st.text_input("⚠️ It seems you're upset. Please provide your email for a quick escalation:")
        if user_email:
            escalation_msg = escalate_to_support(user_email, user_input)
            st.error(escalation_msg)
