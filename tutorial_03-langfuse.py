import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from langfuse_utils import run_llm_call, generate_session_id, create_langfuse_client

# Load environment variables from .env file
load_dotenv()

"""
# Chosen model identifier
model_id = "gpt-4o-mini"

# Configure OpenRouter model
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=model_id,
    temperature=0.7,
    max_tokens=1000,
)
"""

model = AzureChatOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_VERSION_4_1"),  
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_4_1"),
    model_name=os.environ.get("AZURE_OPENAI_DEPLOYMENT_4_1"),
    streaming=False,
    timeout=120.0, 
    max_retries=3,
    verbose=True
)

session_id = generate_session_id()
print(f"Session ID: {session_id}\n")

langfuse_client = create_langfuse_client()
response = run_llm_call(langfuse_client, session_id, model, "What is the square root of 144?")

print(f"\nInput:    What is the square root of 144?")
print(f"Response: {response}")

langfuse_client.flush()

print(f"\n✓ Trace sent to Langfuse with full token usage and cost data")
print(f"✓ Grouped under session: {session_id}")
print("✓ You can inspect this session using get_trace_info(session_id) and print_results(info) below.")