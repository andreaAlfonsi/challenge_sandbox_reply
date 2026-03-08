import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from langfuse_utils import run_llm_call, generate_session_id, create_langfuse_client

# Load environment variables from .env file
load_dotenv()

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

langfuse_client = create_langfuse_client()

questions = [
    "What is machine learning?",
    "Explain neural networks briefly.",
    "What is the difference between AI and ML?"
]

session_id = generate_session_id()
print(f"Session ID: {session_id}")
print(f"Making {len(questions)} agent calls with Langfuse tracing...\n")

for i, question in enumerate(questions, 1):
    response = run_llm_call(langfuse_client, session_id, model, question)
    print(f"Call {i}: {question[:40]}...")
    print(f"  Response: {response[:80]}...\n")

langfuse_client.flush()

print("=" * 50)
print(f"✓ All {len(questions)} traces sent to Langfuse!")
print(f"✓ All grouped under session: {session_id}")
print("✓ You can inspect this session using get_trace_info(session_id) and print_results(info) below.")