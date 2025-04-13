import os
import logging
import asyncio
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
import uuid
from datetime import datetime
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

# Import models and utilities
from app.models.schemas import KnowledgeGraph, GraphNode, GraphEdge
from app.utils.nlp_utils import extract_entities_and_relations

# Set up logging
logger = logging.getLogger(__name__)

# Initialize NLP components
try:
    # Load spaCy model for entity extraction
    _nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy en_core_web_sm model")
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    logger.warning("Trying to download spaCy model...")
    try:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        _nlp = spacy.load("en_core_web_sm")
        logger.info("Successfully downloaded and loaded spaCy model")
    except Exception as inner_e:
        logger.error(f"Failed to download spaCy model: {str(inner_e)}")
        logger.warning("Using simple NLP methods instead of spaCy")
        _nlp = None

# Initialize NLTK components
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading required NLTK data...")
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        logger.info("Successfully downloaded NLTK data")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {str(e)}")

# Get stopwords once for efficiency
try:
    _stop_words = set(stopwords.words('english'))
except Exception:
    _stop_words = set()
    logger.warning("Failed to load stopwords, proceeding with empty stopword list")

# Academic terminology to identify important concepts
ACADEMIC_TERMINOLOGY = {
    "algorithm", "method", "theory", "framework", "model", "approach", "technique",
    "system", "paradigm", "architecture", "process", "protocol", "experiment",
    "analysis", "evaluation", "implementation", "design", "development", "solution",
    "challenge", "problem", "hypothesis", "conclusion", "finding", "result",
    "neural network", "deep learning", "machine learning", "artificial intelligence",
    "transformer", "attention mechanism", "embedding", "classification", "regression",
    "clustering", "optimization", "dataset", "benchmark", "performance", "accuracy",
    "precision", "recall", "f1 score", "loss function", "gradient descent"
}

async def build_knowledge_graph(paper_ids: List[str]) -> KnowledgeGraph:
    """
    Build a knowledge graph from processed papers.
    
    Args:
        paper_ids: List of paper IDs to include in the graph
        
    Returns:
        KnowledgeGraph with nodes and edges
    """
    try:
        # Get paper content from the vector store
        from app.services.process_papers import get_paper_by_id
        
        papers = []
        for paper_id in paper_ids:
            paper = await get_paper_by_id(paper_id)
            if paper:
                papers.append(paper)
        
        if not papers:
            logger.warning(f"No papers found for the provided IDs: {paper_ids}")
            return KnowledgeGraph(nodes=[], edges=[])
        
        # Run graph building in a separate thread to avoid blocking
        return await asyncio.to_thread(
            _build_graph_sync,
            papers
        )
    
    except Exception as e:
        logger.error(f"Error in build_knowledge_graph: {str(e)}", exc_info=True)
        # Return empty graph on error
        return KnowledgeGraph(nodes=[], edges=[])

def _build_graph_sync(papers: List[Dict[str, Any]]) -> KnowledgeGraph:
    """
    Synchronous implementation of building a knowledge graph.
    
    Args:
        papers: List of paper objects with id, content, and metadata
        
    Returns:
        KnowledgeGraph with nodes and edges
    """
    try:
        # Initialize NetworkX graph
        G = nx.DiGraph()
        
        # Keep track of all entities to establish connections between papers
        all_entities = {}  # entity_text -> set of paper_ids
        
        # Process each paper
        for paper in papers:
            paper_id = paper["id"]
            content = paper["content"]
            metadata = paper["metadata"]
            
            logger.info(f"Adding paper to graph: {metadata.get('title', 'Untitled')}")
            
            # Add paper node
            paper_node_id = f"doc_{paper_id}"
            G.add_node(
                paper_node_id,
                id=paper_node_id,
                label="DOCUMENT",
                type="document",
                metadata=metadata
            )
            
            # Process paper content to extract entities and relations
            entities, relations = process_paper_content(content)
            
            # Add entities to graph and connect to paper
            for entity_id, entity_data in entities.items():
                entity_type = entity_data["type"]
                entity_text = entity_data["text"]
                entity_count = entity_data["count"]
                
                # Only add entities that appear enough times (filter out noise)
                min_occurrences = 2 if entity_type == "ACADEMIC_TERM" else 1
                if entity_count < min_occurrences:
                    continue
                
                # Add entity node if it doesn't exist
                if not G.has_node(entity_id):
                    G.add_node(
                        entity_id,
                        id=entity_id,
                        label=entity_type,
                        type="entity",
                        text=entity_text,
                        count=entity_count
                    )
                else:
                    # Update count if entity already exists
                    G.nodes[entity_id]["count"] = G.nodes[entity_id].get("count", 0) + entity_count
                
                # Connect paper to entity
                G.add_edge(
                    paper_node_id,
                    entity_id,
                    id=f"{paper_node_id}_to_{entity_id}",
                    source=paper_node_id,
                    target=entity_id,
                    label="CONTAINS",
                    weight=entity_count
                )
                
                # Track this entity for cross-paper connections
                if entity_text not in all_entities:
                    all_entities[entity_text] = set()
                all_entities[entity_text].add(paper_id)
            
            # Add relations between entities
            for relation in relations:
                source_id = relation["source"]
                target_id = relation["target"]
                relation_type = relation["relation"]
                weight = relation["weight"]
                
                # Only add relations between existing entities
                if G.has_node(source_id) and G.has_node(target_id):
                    # Check if relation already exists
                    if G.has_edge(source_id, target_id):
                        # Update weight if relation exists
                        G[source_id][target_id]["weight"] += weight
                    else:
                        # Add new relation
                        G.add_edge(
                            source_id,
                            target_id,
                            id=f"{source_id}_to_{target_id}",
                            source=source_id,
                            target=target_id,
                            label=relation_type,
                            weight=weight
                        )
        
        # Connect papers that share entities
        connect_papers_with_shared_entities(G, papers, all_entities)
        
        # Convert NetworkX graph to KnowledgeGraph schema
        knowledge_graph = convert_nx_to_schema(G)
        
        logger.info(f"Built knowledge graph with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.edges)} edges")
        return knowledge_graph
    
    except Exception as e:
        logger.error(f"Error in _build_graph_sync: {str(e)}", exc_info=True)
        # Return empty graph on error
        return KnowledgeGraph(nodes=[], edges=[])

def process_paper_content(content: str) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process paper content to extract entities and relations.
    
    Args:
        content: The paper content text
        
    Returns:
        Tuple of (entities_dict, relations_list)
    """
    entities = {}  # entity_id -> {type, text, count}
    relations = []  # List of {source, target, relation, weight}
    
    try:
        # Split into sentences for better context
        sentences = sent_tokenize(content)
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.strip()) < 15:
                continue
            
            # Extract entities and relations using NLP utilities
            sent_entities, sent_relations = extract_entities_and_relations(sentence)
            
            # Process entities
            for entity_text, entity_type in sent_entities:
                # Normalize entity text
                normalized_text = entity_text.lower().strip()
                if not normalized_text or len(normalized_text) < 3:
                    continue
                
                # Create unique ID for this entity
                entity_id = f"{entity_type.lower()}_{normalized_text.replace(' ', '_')}"
                
                # Add or update entity
                if entity_id in entities:
                    entities[entity_id]["count"] += 1
                else:
                    entities[entity_id] = {
                        "type": entity_type,
                        "text": normalized_text,
                        "count": 1
                    }
            
            # Process relations between entities
            for subj, verb, obj in sent_relations:
                # Find the entity IDs
                subj_norm = subj.lower().strip()
                obj_norm = obj.lower().strip()
                
                # Find matching entity IDs
                subj_id = None
                obj_id = None
                
                # Look for exact matches in entities
                for entity_id, entity_data in entities.items():
                    if entity_data["text"] == subj_norm:
                        subj_id = entity_id
                    if entity_data["text"] == obj_norm:
                        obj_id = entity_id
                
                # Skip if both subject and object weren't found
                if not subj_id or not obj_id:
                    continue
                
                # Add relation
                relation = {
                    "source": subj_id,
                    "target": obj_id,
                    "relation": verb.upper().replace(" ", "_"),
                    "weight": 1.0
                }
                
                # Check if relation already exists
                exists = False
                for existing_rel in relations:
                    if (existing_rel["source"] == relation["source"] and 
                        existing_rel["target"] == relation["target"] and
                        existing_rel["relation"] == relation["relation"]):
                        existing_rel["weight"] += 1.0
                        exists = True
                        break
                
                if not exists:
                    relations.append(relation)
        
        # Extract additional academic terminology
        extract_academic_terms(content, entities)
        
        return entities, relations
    
    except Exception as e:
        logger.error(f"Error processing paper content: {str(e)}", exc_info=True)
        return entities, relations

def extract_academic_terms(content: str, entities: Dict[str, Dict[str, Any]]) -> None:
    """
    Extract academic terminology from content and add to entities.
    
    Args:
        content: The paper content
        entities: Dictionary of entities to update
    """
    content_lower = content.lower()
    
    # Look for academic terms in content
    for term in ACADEMIC_TERMINOLOGY:
        if term in content_lower:
            # Create a unique ID for this term
            term_id = f"academic_term_{term.replace(' ', '_')}"
            
            # Count occurrences (simple approach)
            count = content_lower.count(term)
            
            # Only add if it appears enough times
            if count >= 2:
                if term_id in entities:
                    entities[term_id]["count"] += count
                else:
                    entities[term_id] = {
                        "type": "ACADEMIC_TERM",
                        "text": term,
                        "count": count
                    }

def connect_papers_with_shared_entities(G: nx.DiGraph, papers: List[Dict[str, Any]], all_entities: Dict[str, set]) -> None:
    """
    Connect papers that share significant entities.
    
    Args:
        G: The NetworkX graph to update
        papers: List of papers
        all_entities: Dictionary mapping entity text to set of paper IDs
    """
    # Find entities shared between papers
    for entity_text, paper_ids in all_entities.items():
        if len(paper_ids) < 2:
            continue  # Skip entities that appear in only one paper
        
        # Connect papers that share this entity
        paper_list = list(paper_ids)
        for i in range(len(paper_list)):
            for j in range(i+1, len(paper_list)):
                source_paper_id = f"doc_{paper_list[i]}"
                target_paper_id = f"doc_{paper_list[j]}"
                
                # Skip if either paper node doesn't exist
                if not G.has_node(source_paper_id) or not G.has_node(target_paper_id):
                    continue
                
                # Add or update edge
                if G.has_edge(source_paper_id, target_paper_id):
                    G[source_paper_id][target_paper_id]["weight"] += 1.0
                else:
                    G.add_edge(
                        source_paper_id,
                        target_paper_id,
                        id=f"{source_paper_id}_to_{target_paper_id}",
                        source=source_paper_id,
                        target=target_paper_id,
                        label="RELATED_TO",
                        weight=1.0
                    )

def convert_nx_to_schema(G: nx.DiGraph) -> KnowledgeGraph:
    """
    Convert NetworkX graph to KnowledgeGraph schema.
    
    Args:
        G: NetworkX graph
        
    Returns:
        KnowledgeGraph object
    """
    nodes = []
    edges = []
    
    # Convert nodes
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        
        # Handle node type
        node_type = node_data.get("type", "entity")
        
        # Prepare metadata
        metadata = None
        if node_type == "document":
            metadata = node_data.get("metadata", {})
        else:
            # For entity nodes, include text and count in metadata
            metadata = {
                "text": node_data.get("text", ""),
                "count": node_data.get("count", 1)
            }
        
        # Create GraphNode
        node = GraphNode(
            id=node_id,
            label=node_data.get("label", "UNKNOWN"),
            type=node_type,
            metadata=metadata
        )
        nodes.append(node)
    
    # Convert edges
    for source, target, data in G.edges(data=True):
        # Create GraphEdge
        edge = GraphEdge(
            source=source,
            target=target,
            label=data.get("label", "RELATED_TO"),
            weight=data.get("weight", 1.0)
        )
        edges.append(edge)
    
    return KnowledgeGraph(nodes=nodes, edges=edges)
