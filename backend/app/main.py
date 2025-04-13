import os
import logging
from fetch_papers import fetch_papers
from process_papers import process_papers
from build_graph import build_knowledge_graph
from visualize_graph import visualize_graph
from query_agent import create_agent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run ResearchWebGraph."""
    print("Welcome to ResearchWebGraph!")
    
    # Check for API tokens
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN environment variable not set. You'll need to provide this later.")
    
    try:
        # Get user query for papers
        query = input("Enter a topic to fetch research papers (e.g., 'machine learning'): ")
        if not query.strip():
            print("No query provided, using default: 'machine learning'")
            query = "machine learning"
        
        # Fetch papers with error handling
        print(f"Fetching papers on '{query}'...")
        try:
            documents = fetch_papers(query)
            if not documents:
                print("No papers found. Please try a different query.")
                return
            print(f"Successfully fetched {len(documents)} papers")
        except Exception as e:
            print(f"Error fetching papers: {e}")
            return
        
        # Process papers with error handling
        print("Processing papers...")
        try:
            vectorstore = process_papers(documents)
            if vectorstore is None:
                print("Error creating vector store. Check your HF_TOKEN or network connection.")
                return
            print("Vector store created successfully")
        except Exception as e:
            print(f"Error processing papers: {e}")
            return
        
        # Build knowledge graph with error handling
        print("Building knowledge graph...")
        try:
            knowledge_graph = build_knowledge_graph(documents)
            node_count = knowledge_graph.number_of_nodes()
            edge_count = knowledge_graph.number_of_edges()
            print(f"Knowledge graph built with {node_count} nodes and {edge_count} edges")
            
            # Visualize graph
            print("Generating knowledge graph visualization...")
            vis_path = visualize_graph(knowledge_graph)
            print(f"Visualization saved to: {vis_path}")
        except Exception as e:
            print(f"Error building or visualizing knowledge graph: {e}")
            knowledge_graph = None
        
        # Create agent with error handling
        print("Creating query agent...")
        try:
            agent = create_agent(vectorstore, knowledge_graph)
            print("Agent created successfully")
        except Exception as e:
            print(f"Error creating agent: {e}")
            return
        
        # Query loop
        print("\nYou can now ask questions about the papers. Type 'exit' to quit.")
        while True:
            user_input = input("\nQuestion: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            try:
                response = agent.invoke({"input": user_input})
                output = response.get("output", "No response generated")
                print(f"\nAnswer: {output}")
            except Exception as e:
                print(f"Error answering query: {e}")
        
        print("Thank you for using ResearchWebGraph!")
        
    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
