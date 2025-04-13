import spacy
import networkx as nx
import logging
import sys
from typing import List, Dict, Tuple, Any, Optional
from langchain_core.documents import Document
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model with exception handling
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy en_core_web_sm model")
except OSError:
    logger.error("spaCy en_core_web_sm model not found. Attempting to download...")
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
        logger.info("Downloaded and loaded spaCy en_core_web_sm model")
    except Exception as e:
        logger.error(f"Failed to download spaCy model: {e}")
        logger.error("Please install it manually with: python -m spacy download en_core_web_sm")
        # Create a minimal model as fallback
        nlp = spacy.blank("en")
        logger.warning("Using blank English model as fallback - entity recognition will be limited")

# Define academic entity types to extract beyond standard spaCy entities
ACADEMIC_KEYWORDS = [
    "algorithm", "method", "framework", "model", "architecture", "system",
    "theory", "theorem", "hypothesis", "analysis", "experiment", "study",
    "dataset", "benchmark", "result", "performance", "accuracy", "neural",
    "deep learning", "machine learning", "artificial intelligence", "AI",
    "NLP", "computer vision", "reinforcement learning", "network", "transformer"
]

def extract_entities_and_relations(text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    Extract entities and relationships from text using spaCy with academic focus.
    
    Args:
        text: The text to process
        
    Returns:
        Tuple containing (entities, relations)
        - entities: List of (entity_text, entity_type) tuples
        - relations: List of (subject, predicate, object) tuples
    """
    if not text or not text.strip():
        return [], []
    
    try:
        # Process with spaCy
        doc = nlp(text)
        
        # Extract named entities
        standard_entities = [
            (ent.text, ent.label_) 
            for ent in doc.ents 
            if ent.label_ in ["PERSON", "ORG", "GPE", "NORP", "WORK_OF_ART", "DATE", "PRODUCT"]
        ]
        
        # Extract academic terms using noun chunks and keywords
        academic_entities = []
        for chunk in doc.noun_chunks:
            # Clean up the chunk text
            chunk_text = chunk.text.strip().lower()
            
            # Check if the chunk contains academic keywords
            if any(keyword in chunk_text for keyword in ACADEMIC_KEYWORDS):
                academic_entities.append((chunk.text, "ACADEMIC_TERM"))
            
            # Check if the chunk is a potential technical term (>1 word, contains no stopwords)
            elif (len(chunk_text.split()) > 1 and 
                  not all(token.is_stop for token in chunk)):
                academic_entities.append((chunk.text, "TECH_TERM"))
        
        # Combine all entities
        entities = standard_entities + academic_entities
        
        # Extract relations (subject-verb-object triples)
        relations = []
        for sent in doc.sents:
            # Find verbs and their dependencies
            for token in sent:
                # Focus on verbs with both subject and object
                if token.pos_ == "VERB":
                    subj = None
                    obj = None
                    
                    # Find subject and object
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            # Get the full subject phrase
                            subj_span = get_span_for_token(child)
                            if subj_span:
                                subj = subj_span.text
                        elif child.dep_ in ["dobj", "pobj"]:
                            # Get the full object phrase
                            obj_span = get_span_for_token(child)
                            if obj_span:
                                obj = obj_span.text
                    
                    # Only add complete relations
                    if subj and obj:
                        relations.append((subj, token.lemma_, obj))
        
        return entities, relations
    
    except Exception as e:
        logger.error(f"Error extracting entities and relations: {str(e)}")
        return [], []

def get_span_for_token(token):
    """
    Get the full noun phrase span for a token by finding its head and subtree.
    """
    # If token is part of a noun phrase, find the root of the phrase
    if token.pos_ in ["NOUN", "PROPN"]:
        # Find the head of the noun phrase
        head = token
        while head.head.pos_ in ["NOUN", "PROPN"] and head.head != head:
            head = head.head
            
        # Get the full span by finding the leftmost and rightmost tokens in the subtree
        subtree = list(head.subtree)
        start = min(subtree, key=lambda t: t.i)
        end = max(subtree, key=lambda t: t.i)
        return token.doc[start.i:end.i+1]
    else:
        # For non-nouns, just return a span of the token itself
        return token.doc[token.i:token.i+1]

def build_knowledge_graph(documents: List[Document]) -> nx.DiGraph:
    """
    Build a knowledge graph from documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        NetworkX directed graph with entities as nodes and relationships as edges
    """
    # Create a directed graph for proper relationship representation
    G = nx.DiGraph()
    
    if not documents:
        logger.warning("No documents provided for knowledge graph construction")
        return G
    
    try:
        # Process each document
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)} for knowledge graph")
            
            # Get content and metadata
            content = doc.page_content
            metadata = doc.metadata
            doc_id = metadata.get("arxiv_id", f"doc_{i}")
            doc_title = metadata.get("title", f"Document {i}")
            
            # Add document node
            G.add_node(
                doc_title, 
                label="DOCUMENT",
                type="document",
                id=doc_id,
                metadata=metadata
            )
            
            # Extract entities and relations
            entities, relations = extract_entities_and_relations(content)
            
            # Add entity nodes with tracking to avoid duplicates
            entity_nodes = {}
            for entity_text, entity_type in entities:
                # Skip very short entities
                if len(entity_text.strip()) < 3:
                    continue
                    
                # Create unique node key
                clean_text = entity_text.strip()
                node_key = f"{clean_text.lower()}_{entity_type}"
                
                if node_key not in entity_nodes:
                    # Add new node
                    G.add_node(
                        clean_text,
                        label=entity_type,
                        type="entity"
                    )
                    entity_nodes[node_key] = clean_text
                    
                    # Connect entity to its source document
                    G.add_edge(
                        doc_title,
                        clean_text,
                        label="CONTAINS",
                        weight=1
                    )
                else:
                    # Entity already exists, increment document connection weight
                    if G.has_edge(doc_title, entity_nodes[node_key]):
                        G[doc_title][entity_nodes[node_key]]["weight"] += 1
                    else:
                        G.add_edge(
                            doc_title, 
                            entity_nodes[node_key],
                            label="CONTAINS",
                            weight=1
                        )
            
            # Add relations as edges between entities
            for subj, verb, obj in relations:
                # Find matching entity nodes for subject and object
                subj_nodes = [
                    node for node in G.nodes() 
                    if isinstance(node, str) and subj.lower() in node.lower()
                ]
                obj_nodes = [
                    node for node in G.nodes() 
                    if isinstance(node, str) and obj.lower() in node.lower()
                ]
                
                # Add edges between matching entities
                for s_node in subj_nodes:
                    for o_node in obj_nodes:
                        if s_node != o_node:  # Avoid self-loops
                            # Check if edge exists
                            if G.has_edge(s_node, o_node):
                                # Increment weight for existing edge
                                G[s_node][o_node]["weight"] += 1
                                
                                # Add this verb to the verb list if not present
                                if "verbs" in G[s_node][o_node]:
                                    if verb not in G[s_node][o_node]["verbs"]:
                                        G[s_node][o_node]["verbs"].append(verb)
                                else:
                                    G[s_node][o_node]["verbs"] = [verb]
                                    
                                # Update the primary label to the most common verb
                                most_common_verb = max(
                                    G[s_node][o_node]["verbs"], 
                                    key=G[s_node][o_node]["verbs"].count
                                )
                                G[s_node][o_node]["label"] = most_common_verb.upper()
                            else:
                                # Create new edge
                                G.add_edge(
                                    s_node, 
                                    o_node,
                                    label=verb.upper(),
                                    verbs=[verb],
                                    weight=1,
                                    doc_source=doc_id
                                )
        
        logger.info(f"Knowledge graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
        
    except Exception as e:
        logger.error(f"Error building knowledge graph: {str(e)}")
        # Return whatever we've built so far
        return G
