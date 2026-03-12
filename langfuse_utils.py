import os
from dotenv import load_dotenv

import ulid
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage

load_dotenv()

def create_langfuse_client():
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
    )

def generate_session_id():
    """Generate a unique session ID using TEAM_NAME and ULID."""
    return f"{os.getenv('TEAM_NAME', 'tutorial')}-{ulid.new().str}"

def invoke_langchain(model, prompt, langfuse_handler):
    """Invoke LangChain with the given prompt and Langfuse handler."""
    response = model.invoke(prompt, config={"callbacks": [langfuse_handler]})
    return response

@observe()
def run_llm_call(langfuse_client, session_id, model, prompt):
    """Run a single LangChain invocation and track it in Langfuse."""
    # Update trace with session_id
    langfuse_client.update_current_trace(session_id=session_id)

    # Create Langfuse callback handler for automatic generation tracking
    # The handler will attach to the current trace created by @observe()
    langfuse_handler = CallbackHandler()

    # Invoke LangChain with Langfuse handler to track tokens and costs
    response = invoke_langchain(model, prompt, langfuse_handler)

    return response
