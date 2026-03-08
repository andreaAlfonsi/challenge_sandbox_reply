import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

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

@tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert temperature between Celsius and Fahrenheit.
    
    Args:
        value: The temperature value to convert
        from_unit: Source unit ("celsius" or "fahrenheit")
        to_unit: Target unit ("celsius" or "fahrenheit")
    
    Returns:
        Converted temperature value
    """
    if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        return (value * 9/5) + 32
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        return (value - 32) * 5/9
    elif from_unit.lower() == to_unit.lower():
        return value
    else:
        raise ValueError(f"Unsupported conversion from {from_unit} to {to_unit}")

agent = create_agent(
    model=model,
    system_prompt="You are a helpful assistant that can convert temperatures between Celsius and Fahrenheit. Always use the convert_temperature tool when users ask for temperature conversions.",
    tools=[convert_temperature]
)

# Test 1: Simple conversion
response = agent.invoke({"messages": [HumanMessage("Convert 25 degrees Celsius to Fahrenheit")]})
print(response["messages"][-1].content)

# Test 2: Reverse conversion
response = agent.invoke({"messages": [HumanMessage("What is 77 degrees Fahrenheit in Celsius?")]})
print(response["messages"][-1].content)

# Test 3: More complex question
response = agent.invoke({"messages": [HumanMessage("If it's 20°C outside, what would that be in Fahrenheit? Is that warm or cold?")]})
print(response["messages"][-1].content)