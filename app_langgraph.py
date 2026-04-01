# LangGraph + Agentic AI: Customer Support Agent with Multiple Tools + Streamlit UI using LangGraph
# -------------------------------------------------------------------------------
# Tools: FAQ retrieval, Order Status Checker (via CSV), Sentiment Analyzer (LLM), Email Escalation, General Chatbot

import pandas as pd
import streamlit as st
import smtplib
import os
import re
from dotenv import load_dotenv
from email.mime.text import MIMEText

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0.1)

# Load FAQs into vector DB
loader = PyPDFLoader("Customer_Support_FAQs.pdf")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = splitter.split_documents(documents)
faq_vector_db = FAISS.from_documents(docs, GoogleGenerativeAIEmbeddings(model="models/embedding-gecko-001", google_api_key=google_api_key))

# Tool implementations
def check_order_status(order_id: str) -> str:
    df = pd.read_csv("order_data.csv")
    order = df[df['order_id'] == order_id]
    if order.empty:
        return "Order ID not found. Please check and try again."
    row = order.iloc[0]
    return f"Order Status for {order_id}: {row['status']}, Estimated Delivery: {row['eta']}"

def get_faq_response(query: str) -> str:
    return faq_vector_db.similarity_search(query, k=1)[0].page_content

def general_chat_response(message: str) -> str:
    prompt = f"You are a helpful and friendly customer support assistant. Answer briefly and clearly:\n{message}"
    result = llm.invoke(prompt)
    return result.content.strip()

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

def analyze_sentiment(text: str) -> str:
    prompt = PromptTemplate.from_template("""
    Analyze the sentiment of the following customer message:
    "{message}"
    Is it Positive, Neutral, or Negative?
    Just reply with one word: Positive, Neutral, or Negative.
    """)
    result = llm.invoke(prompt.format(message=text))
    return result.content.strip().lower()

# Define tools
tool_executor = ToolExecutor([
    {
        "name": "CheckOrderStatus",
        "description": "Use to check the status of an order using Order ID",
        "func": lambda x: check_order_status(x)
    },
    {
        "name": "FAQTool",
        "description": "Use to retrieve answers to common FAQs.",
        "func": lambda x: get_faq_response(x)
    },
    {
        "name": "GeneralChat",
        "description": "Use to respond in general conversation or when no other tool is suitable.",
        "func": lambda x: general_chat_response(x)
    }
])

# Graph State and Router
class AgentState(dict):
    input: str
    tool_invocation: ToolInvocation
    tool_response: str

class Router(Runnable):
    def invoke(self, input: AgentState, **kwargs) -> str:
        text = input["input"].lower()
        if re.search(r"\b\d{5}\b", text):
            return "CheckOrderStatus"
        elif any(word in text for word in ["return", "warranty", "payment", "track", "cancel"]):
            return "FAQTool"
        else:
            return "GeneralChat"

# LangGraph Workflow
def create_graph():
    graph = StateGraph(AgentState)
    graph.add_node("CheckOrderStatus", lambda state: {"tool_response": tool_executor.invoke(ToolInvocation(tool="CheckOrderStatus", input=state["input"]))})
    graph.add_node("FAQTool", lambda state: {"tool_response": tool_executor.invoke(ToolInvocation(tool="FAQTool", input=state["input"]))})
    graph.add_node("GeneralChat", lambda state: {"tool_response": tool_executor.invoke(ToolInvocation(tool="GeneralChat", input=state["input"]))})
    graph.add_conditional_edges("entry", Router(), {
        "CheckOrderStatus": "CheckOrderStatus",
        "FAQTool": "FAQTool",
        "GeneralChat": "GeneralChat"
    })
    graph.add_edge("CheckOrderStatus", END)
    graph.add_edge("FAQTool", END)
    graph.add_edge("GeneralChat", END)
    return graph.compile()

workflow = create_graph()

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

    result = workflow.invoke({"input": user_input})
    st.write(result["tool_response"])

    if sentiment == "negative":
        user_email = st.text_input("⚠️ It seems you're upset. Please provide your email for a quick escalation:")
        if user_email:
            escalation_msg = escalate_to_support(user_email, user_input)
            st.error(escalation_msg)
