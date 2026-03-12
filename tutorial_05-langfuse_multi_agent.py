import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
import loguru
from langfuse_utils import run_llm_call, generate_session_id, create_langfuse_client

logger = loguru.logger

session_id = generate_session_id()
print(f"Session ID: {session_id}\n")

langfuse_client = create_langfuse_client()

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

# Create specialized logistics planning agent
logistics_agent = create_agent(
    model=model,
    system_prompt="""You are a travel logistics expert. You handle practical travel planning:
    - Calculate distances between locations and travel times
    - Estimate costs for transportation, accommodation, and activities
    - Optimize routes and suggest efficient itineraries
    - Consider time zones, weather, and practical constraints
    Always provide short, clear, practical logistics information."""
)

# Create specialized recommendations agent
recommendations_agent = create_agent(
    model=model,
    system_prompt="""You are a travel recommendations specialist. You suggest experiences and activities:
    - Recommend top attractions, landmarks, and must-see places
    - Suggest restaurants, local cuisine, and dining experiences
    - Recommend cultural activities, events, and local experiences
    - Provide insights about local customs, best times to visit, and hidden gems
    Always provide brief, engaging, personalized recommendations."""
)

@tool
def plan_logistics_agent(trip_request: str) -> str:
    """
    Plan travel logistics including distances, times, costs, and routes.
    Use this to calculate practical travel information and optimize itineraries.
    
    Args:
        trip_request: Trip details (e.g., "3 days in Paris, budget $1500, from London")
    
    Returns:
        Logistics information: distances, travel times, costs, and route suggestions
    """
    logger.info("Tool called: plan_logistics_agent with request: {}", trip_request)
    response = logistics_agent.invoke({"messages": [HumanMessage(f"Plan logistics for this trip: {trip_request}")]})
    return response["messages"][-1].content

@tool
def get_recommendations_agent(trip_details: str) -> str:
    """
    Get travel recommendations for attractions, restaurants, and activities.
    Use this to suggest what to see, do, and eat at the destination.
    
    Args:
        trip_details: Destination and trip information (e.g., "3 days in Paris, interested in art and food")
    
    Returns:
        Recommendations for attractions, restaurants, activities, and cultural insights
    """
    logger.info("Tool called: get_recommendations_agent with details: {}", trip_details)
    response = recommendations_agent.invoke({"messages": [HumanMessage(f"Provide recommendations for: {trip_details}")]})
    return response["messages"][-1].content
    
orchestrator = create_agent(
    model=model,
    system_prompt="""You are a travel planning coordinator. 
    When planning trips, use both specialists:
    1. Use plan_logistics_agent to calculate practical details: distances, times, costs, and routes
    2. Use get_recommendations_agent to suggest attractions, restaurants, and activities
    Always combine both the practical logistics and exciting recommendations in your final response.""",
    tools=[plan_logistics_agent, get_recommendations_agent]
)

# Test 1: City trip planning - uses both agents
response = run_llm_call(langfuse_client, session_id, orchestrator, {"messages": [HumanMessage("Plan a 3-day trip to Rome. I'm coming from London with a budget of $2000. Calculate travel costs and time, and suggest must-see attractions and restaurants.")]})
print(f"Response: {response["messages"][-1].content}")

# Test 2: Multi-city itinerary - both agents work together
response = run_llm_call(langfuse_client, session_id, orchestrator, {"messages": [HumanMessage("I want to visit Paris, Amsterdam, and Berlin in 7 days starting from New York. Plan the logistics (flights, trains, costs) and recommend top 3 things to do in each city.")]})
print(f"Response: {response["messages"][-1].content}")

# Test 3: Weekend getaway - comprehensive planning
response = run_llm_call(langfuse_client, session_id, orchestrator, {"messages": [HumanMessage("Plan a weekend trip to Barcelona from Madrid. Budget is $500. Calculate travel time and costs, and suggest the best places to visit, eat, and experience local culture.")]})
print(f"Response: {response["messages"][-1].content}")