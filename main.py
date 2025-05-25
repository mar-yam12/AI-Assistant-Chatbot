import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from agents import Agent, RunConfig, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
import asyncio

# Load environment variables
load_dotenv(find_dotenv())

# Get the Gemini API Key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY is not set in your .env file.")
    st.stop()

# Step 1: Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Step 2: Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

# Step 3: Config
config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# Step 4: Agent
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    model=model
)

# Streamlit UI
st.set_page_config(page_title="AI Assistant Chatbot", page_icon="💬")
st.title("🤖 Maryam AI Assistant")

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Interface
user_input = st.chat_input("Type your message...")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the agent using asyncio
    async def get_response():
        return await Runner.run(
            agent,
            input=st.session_state.chat_history,
            run_config=config
        )

    result = asyncio.run(get_response())
    assistant_reply = result.final_output

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
