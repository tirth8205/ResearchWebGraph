import os
import logging
from typing import Optional, List, Dict, Any
import importlib.util

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for required dependencies
required_packages = ["langchain", "langchain_huggingface", "langchain_core"]
missing_packages = []
for package in required_packages:
    if importlib.util.find_spec(package) is None:
        missing_packages.append(package)
        
if missing_packages:
    error_msg = f"Missing required packages: {', '.join(missing_packages)}. Please install them."
    logger.error(error_msg)
    raise ImportError(error_msg)

# Import LangChain components with up-to-date paths
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import AgentExecutor, create_react_agent

def create_agent(vectorstore, knowledge_graph, api_key: Optional[str] = None, 
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Create an agent for querying papers and the knowledge graph.
    
    Args:
        vectorstore: FAISS vector store containing processed papers
        knowledge_graph: NetworkX graph of entities and relationships
        api_key: HuggingFace API token (optional if set as environment variable)
        model_name: HuggingFace model name to use for the agent
        
    Returns:
        AgentExecutor object capable of answering questions about papers
    """
    # Validate inputs
    if vectorstore is None:
        logger.warning("No vector store provided, agent will have limited functionality")
    
    if knowledge_graph is None or knowledge_graph.number_of_nodes() == 0:
        logger.warning("Empty knowledge graph provided, agent will have limited functionality")
    
    # Get API token with validation
    hf_token = api_key or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "No Hugging Face API token found. Please set the HF_TOKEN environment variable "
            "or provide the api_key parameter."
        )
    
    try:
        # Initialize Hugging Face LLM with a more suitable model
        logger.info(f"Initializing HuggingFaceEndpoint with model: {model_name}")
        llm = HuggingFaceEndpoint(
            endpoint_url=f"https://api-inference.huggingface.co/models/{model_name}",
            huggingfacehub_api_token=hf_token,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
        
        # Define tools with clearer descriptions and better error handling
        tools = []
        
        # Add Paper Search tool if vectorstore is available
        if vectorstore:
            def search_papers(query: str) -> str:
                """Search for relevant paper content based on a query."""
                try:
                    if not query or query.strip() == "":
                        return "Please provide a specific search term or question about the papers."
                    
                    results = vectorstore.similarity_search(query, k=3)
                    if not results:
                        return "No relevant information found in the papers."
                    
                    formatted_results = []
                    for i, doc in enumerate(results, 1):
                        metadata = doc.metadata
                        title = metadata.get("title", "Untitled")
                        authors = metadata.get("authors", [])
                        authors_str = ", ".join(authors) if authors else "Unknown"
                        
                        formatted_results.append(
                            f"Document {i}:\n"
                            f"Title: {title}\n"
                            f"Authors: {authors_str}\n"
                            f"Content: {doc.page_content}\n"
                        )
                    
                    return "\n\n".join(formatted_results)
                except Exception as e:
                    logger.error(f"Error in paper search: {str(e)}")
                    return f"Error searching papers: {str(e)}"
            
            tools.append(
                Tool(
                    name="PaperSearch",
                    func=search_papers,
                    description="Use this to search for specific information within research papers. "
                                "Provide a specific question or keywords to find relevant content. "
                                "Returns excerpts from up to 3 most relevant papers."
                )
            )
        
        # Add Knowledge Graph Query tool if graph is available
        if knowledge_graph and knowledge_graph.number_of_nodes() > 0:
            def query_graph(query: str) -> str:
                """Query the knowledge graph for entities and relationships."""
                try:
                    if not query or query.strip() == "":
                        return "Please provide a specific query about entities or relationships."
                    
                    # Parse the query to determine what to return
                    query_lower = query.lower()
                    
                    # Handle specific query types
                    if "related to" in query_lower or "connected to" in query_lower:
                        # Extract entity name from query
                        for node in knowledge_graph.nodes():
                            if isinstance(node, str) and node.lower() in query_lower:
                                # Get all neighbors
                                neighbors = list(knowledge_graph.neighbors(node))
                                if not neighbors:
                                    return f"No entities found related to '{node}'."
                                
                                # Format neighbor information
                                relations = []
                                for neighbor in neighbors:
                                    if knowledge_graph.has_edge(node, neighbor):
                                        edge_data = knowledge_graph.get_edge_data(node, neighbor)
                                        relation = edge_data.get('label', 'RELATED_TO')
                                        relations.append(f"- {node} {relation} {neighbor}")
                                    elif knowledge_graph.has_edge(neighbor, node):
                                        edge_data = knowledge_graph.get_edge_data(neighbor, node)
                                        relation = edge_data.get('label', 'RELATED_TO')
                                        relations.append(f"- {neighbor} {relation} {node}")
                                
                                return f"Relationships for '{node}':\n" + "\n".join(relations)
                    
                    elif "entities" in query_lower or "list" in query_lower:
                        # Group entities by type
                        entity_types = {}
                        for node, data in knowledge_graph.nodes(data=True):
                            entity_type = data.get('label', 'Unknown')
                            if entity_type not in entity_types:
                                entity_types[entity_type] = []
                            entity_types[entity_type].append(node)
                        
                        # Format the response
                        response = ["Entities found in the papers:"]
                        for entity_type, entities in entity_types.items():
                            response.append(f"\n{entity_type} ({len(entities)}):")
                            # Limit to 10 entities per type to avoid very long responses
                            for entity in sorted(entities)[:10]:
                                response.append(f"- {entity}")
                            if len(entities) > 10:
                                response.append(f"- ... and {len(entities) - 10} more")
                                
                        return "\n".join(response)
                    
                    # If query contains "summarize" or "summary", provide paper summaries
                    elif "summarize" in query_lower or "summary" in query_lower or "summarise" in query_lower:
                        # Get document nodes
                        doc_nodes = [node for node, data in knowledge_graph.nodes(data=True) 
                                    if data.get('type') == 'document']
                        
                        if not doc_nodes:
                            return "No paper documents found in the knowledge graph."
                        
                        summaries = []
                        for doc_node in doc_nodes[:3]:  # Limit to 3 papers
                            node_data = knowledge_graph.nodes[doc_node]
                            metadata = node_data.get('metadata', {})
                            title = metadata.get('title', 'Untitled')
                            authors = metadata.get('authors', [])
                            authors_str = ", ".join(authors[:3])
                            if len(authors) > 3:
                                authors_str += f" and {len(authors) - 3} more"
                            
                            # Get connected entities
                            connected_entities = []
                            for neighbor in knowledge_graph.neighbors(doc_node):
                                node_type = knowledge_graph.nodes[neighbor].get('label', 'Unknown')
                                if node_type in ["ACADEMIC_TERM", "TECH_TERM", "ORG"]:
                                    connected_entities.append(neighbor)
                            
                            # Format the summary
                            summary = (
                                f"Paper: {title}\n"
                                f"Authors: {authors_str}\n"
                                f"Key concepts: {', '.join(connected_entities[:5])}\n"
                            )
                            summaries.append(summary)
                        
                        return "Paper Summaries:\n\n" + "\n\n".join(summaries)
                    
                    # Default response - show graph statistics and sample entities
                    response = [
                        f"Knowledge Graph Statistics:",
                        f"- {knowledge_graph.number_of_nodes()} entities",
                        f"- {knowledge_graph.number_of_edges()} relationships",
                        "\nTop entities by connections:"
                    ]
                    
                    # Get top entities by degree
                    top_entities = sorted(
                        [(node, knowledge_graph.degree(node)) for node in knowledge_graph.nodes()],
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                    
                    for entity, degree in top_entities:
                        entity_type = knowledge_graph.nodes[entity].get('label', 'Unknown')
                        response.append(f"- {entity} ({entity_type}): {degree} connections")
                    
                    response.append("\nSuggested queries:")
                    response.append("- 'Show entities related to [specific entity]'")
                    response.append("- 'List all entities'")
                    response.append("- 'Summarize the papers'")
                    
                    return "\n".join(response)
                    
                except Exception as e:
                    logger.error(f"Error querying knowledge graph: {str(e)}")
                    return f"Error querying knowledge graph: {str(e)}"
            
            tools.append(
                Tool(
                    name="KnowledgeGraph",
                    func=query_graph,
                    description="Use this to explore entities (people, organizations, concepts) and "
                                "their relationships extracted from the papers. You can ask for all entities, "
                                "entities related to a specific topic, or connections between entities."
                )
            )
        
        # IMPROVED: Create the prompt template with required variables for React agent and clearer instructions
        template = """You are a research assistant that helps analyze academic papers.

You have access to the following tools:
{tools}

When answering:
- Be concise and accurate
- Cite specific papers when providing information
- Provide context for any entities or relationships you mention
- If you don't know something, say so rather than making up information

USE EXACTLY THIS FORMAT:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (DO NOT include the tool name or parentheses here)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

For example:
Question: What are the main topics in the papers?
Thought: I should search for the main topics in the papers
Action: PaperSearch
Action Input: main topics
Observation: (Result would appear here)
Thought: I now know the main topics
Final Answer: The main topics in the papers are...

Begin!
Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        
        # Create the ReAct agent
        agent = create_react_agent(llm, tools, prompt)
        
        # Define a function to detect and stop loops
        def should_continue(self, iterations, time_elapsed, intermediate_steps):
            # Check if the last 3 steps used the same tool with the same input
            if len(intermediate_steps) >= 3:
                last_three = intermediate_steps[-3:]
                tools_used = [step[0].tool for step in last_three]
                inputs_used = [step[0].tool_input for step in last_three]
                
                # If same tool and same input pattern detected, stop early
                if (tools_used[0] == tools_used[1] == tools_used[2] and
                    inputs_used[0] == inputs_used[1] == inputs_used[2]):
                    logger.warning("Agent loop detected - stopping early")
                    return False
            
            # Continue if no loop detected and under limits
            return iterations < self.max_iterations
        
        # Create the agent executor with increased iterations and better error handling
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            max_iterations=10,  # Increased from default 5
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        # Add the early stopping function
        #setattr(agent_executor, "should_continue", 
        #        lambda iterations, time_elapsed, intermediate_steps: 
        #        should_continue(agent_executor, iterations, time_elapsed, intermediate_steps))
        
        logger.info("Successfully created query agent")
        return agent_executor
    
    except Exception as e:
        logger.error(f"Failed to create agent: {str(e)}")
        raise
