import os
import logging
import re
from typing import List, Tuple, Dict, Any
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from string import punctuation

# Set up logging
logger = logging.getLogger(__name__)

# Initialize NLP components - with fallbacks if packages aren't available
try:
    # Load SpaCy model
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    logger.info("Loaded spaCy en_core_web_sm model")
except Exception as e:
    logger.warning(f"Failed to load spaCy model: {str(e)}")
    SPACY_AVAILABLE = False
    
    # Try to download the model
    try:
        if spacy.util.is_package("en_core_web_sm"):
            logger.info("SpaCy model found but couldn't be loaded. Will try to download.")
        else:
            logger.info("SpaCy model not found. Attempting to download...")
        spacy.cli.download("en_core_web_sm")
        _nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        logger.info("Successfully downloaded and loaded spaCy model")
    except Exception as inner_e:
        logger.warning(f"Failed to download spaCy model: {str(inner_e)}")
        _nlp = None

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    NLTK_TOKENIZER_AVAILABLE = True
except LookupError:
    logger.info("NLTK punkt tokenizer not found. Downloading...")
    try:
        nltk.download('punkt')
        NLTK_TOKENIZER_AVAILABLE = True
        logger.info("Successfully downloaded NLTK punkt tokenizer")
    except Exception as e:
        logger.warning(f"Failed to download NLTK punkt tokenizer: {str(e)}")
        NLTK_TOKENIZER_AVAILABLE = False

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
    NLTK_TAGGER_AVAILABLE = True
except LookupError:
    logger.info("NLTK tagger not found. Downloading...")
    try:
        nltk.download('averaged_perceptron_tagger')
        NLTK_TAGGER_AVAILABLE = True
        logger.info("Successfully downloaded NLTK tagger")
    except Exception as e:
        logger.warning(f"Failed to download NLTK tagger: {str(e)}")
        NLTK_TAGGER_AVAILABLE = False

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
    NLTK_NER_AVAILABLE = True
except LookupError:
    logger.info("NLTK NER chunker not found. Downloading...")
    try:
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        NLTK_NER_AVAILABLE = True
        logger.info("Successfully downloaded NLTK chunker and words")
    except Exception as e:
        logger.warning(f"Failed to download NLTK chunker or words: {str(e)}")
        NLTK_NER_AVAILABLE = False

try:
    nltk.data.find('corpora/stopwords')
    _stop_words = set(stopwords.words('english'))
    NLTK_STOPWORDS_AVAILABLE = True
except LookupError:
    logger.info("NLTK stopwords not found. Downloading...")
    try:
        nltk.download('stopwords')
        _stop_words = set(stopwords.words('english'))
        NLTK_STOPWORDS_AVAILABLE = True
        logger.info("Successfully downloaded NLTK stopwords")
    except Exception as e:
        logger.warning(f"Failed to download NLTK stopwords: {str(e)}")
        NLTK_STOPWORDS_AVAILABLE = False
        _stop_words = set()

# Academic terminology dictionary
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

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks of approximately equal size.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if not text or not isinstance(text, str):
        logger.warning("Invalid text provided for chunking")
        return []
    
    try:
        # Clean the text
        text = text.strip()
        
        # Handle very short texts
        if len(text) <= chunk_size:
            return [text]
        
        # Use sentence tokenization to avoid splitting in the middle of sentences
        if NLTK_TOKENIZER_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            # Fallback using regex
            sentence_endings = r'(?<=[.!?])\s+'
            sentences = re.split(sentence_endings, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed the chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Add the current chunk to the list
                chunks.append(current_chunk)
                
                # Calculate the overlap
                if chunk_overlap > 0:
                    # Start a new chunk with overlap
                    words = current_chunk.split()
                    overlap_word_count = min(len(words), int(chunk_overlap / 10))  # Approximate word count
                    current_chunk = " ".join(words[-overlap_word_count:]) + " " + sentence
                else:
                    # Start a new chunk without overlap
                    current_chunk = sentence
            else:
                # Add the sentence to the current chunk
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"Split text into {len(chunks)} chunks (avg size: {sum(len(c) for c in chunks)/len(chunks):.0f} chars)")
        return chunks
    
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {str(e)}", exc_info=True)
        # Return the whole text as a single chunk if an error occurs
        return [text] if text else []

def extract_entities_and_relations(text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    Extract entities and relationships from text.
    
    Args:
        text: Text to process
        
    Returns:
        Tuple of (entities, relations)
        - entities: List of (entity_text, entity_type) tuples
        - relations: List of (subject, predicate, object) tuples
    """
    entities = []
    relations = []
    
    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return entities, relations
    
    try:
        # Choose strategy based on available NLP tools
        if SPACY_AVAILABLE:
            entities, relations = _extract_with_spacy(text)
        elif NLTK_NER_AVAILABLE and NLTK_TAGGER_AVAILABLE:
            entities, relations = _extract_with_nltk(text)
        else:
            # Simple regex-based extraction as fallback
            entities, relations = _extract_with_regex(text)
        
        # Add academic terms based on terminology dictionary
        academic_terms = _extract_academic_terms(text)
        entities.extend(academic_terms)
        
        return entities, relations
    
    except Exception as e:
        logger.error(f"Error extracting entities and relations: {str(e)}", exc_info=True)
        return [], []

def _extract_with_spacy(text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    Extract entities and relations using spaCy.
    
    Args:
        text: Text to process
        
    Returns:
        Tuple of (entities, relations)
    """
    entities = []
    relations = []
    
    # Process text with spaCy
    doc = _nlp(text)
    
    # Extract named entities
    for ent in doc.ents:
        entity_text = ent.text.strip()
        entity_type = ent.label_
        
        # Skip very short entities and stopwords
        if len(entity_text) < 3 or entity_text.lower() in _stop_words:
            continue
        
        entities.append((entity_text, entity_type))
    
    # Extract subject-verb-object patterns for relations
    for sent in doc.sents:
        # Find root verb
        root = None
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                root = token
                break
        
        if not root:
            continue
        
        # Find subject and object connected to the root
        subject = None
        direct_object = None
        
        for token in sent:
            # Find subject
            if token.dep_ in ("nsubj", "nsubjpass") and token.head == root:
                # Expand to noun phrase if possible
                subject_span = _get_span_for_token(token, sent)
                subject = subject_span.text if subject_span else token.text
            
            # Find object
            if token.dep_ in ("dobj", "pobj") and (token.head == root or token.head.head == root):
                # Expand to noun phrase if possible
                object_span = _get_span_for_token(token, sent)
                direct_object = object_span.text if object_span else token.text
        
        # If we found both subject and object, add the relation
        if subject and direct_object:
            # Skip if either is a pronoun
            if not any(pronoun in subject.lower() for pronoun in ["i", "you", "he", "she", "it", "we", "they"]):
                relations.append((subject, root.lemma_, direct_object))
    
    return entities, relations

def _get_span_for_token(token, sent):
    """
    Get the noun phrase span for a token.
    
    Args:
        token: The token to get a span for
        sent: The sentence containing the token
        
    Returns:
        Span object or None
    """
    # Start with the token itself
    start, end = token.i, token.i + 1
    
    # Expand left to include modifiers
    i = token.i - 1
    while i >= sent.start:
        curr_token = token.doc[i]
        if curr_token.dep_ in ("compound", "amod", "det") and curr_token.head == token.doc[start]:
            start = i
        elif curr_token.dep_ == "punct":
            start = i
        else:
            break
        i -= 1
    
    # Expand right to include modifiers
    i = token.i + 1
    while i < sent.end:
        curr_token = token.doc[i]
        if curr_token.dep_ in ("prep", "pobj", "compound") and curr_token.head == token.doc[end-1]:
            end = i + 1
        elif curr_token.dep_ == "punct":
            end = i + 1
        else:
            break
        i += 1
    
    return token.doc[start:end] if end > start else None

def _extract_with_nltk(text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    Extract entities and relations using NLTK.
    
    Args:
        text: Text to process
        
    Returns:
        Tuple of (entities, relations)
    """
    entities = []
    relations = []
    
    try:
        # Split into sentences
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Tokenize and tag the sentence
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            
            # Named entity recognition
            chunks = ne_chunk(tagged)
            
            # Extract named entities
            current_entity = []
            current_type = None
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    # We're in a named entity
                    entity_text = ' '.join([token for token, pos in chunk.leaves()])
                    entity_type = chunk.label()
                    
                    # Skip very short entities and stopwords
                    if len(entity_text) < 3 or entity_text.lower() in _stop_words:
                        continue
                    
                    entities.append((entity_text, entity_type))
            
            # Extract subject-verb-object patterns
            # This is a simple approach - more complex patterns would require dependency parsing
            nouns = []
            verbs = []
            
            for i, (token, tag) in enumerate(tagged):
                if tag.startswith('NN'):
                    nouns.append((i, token))
                elif tag.startswith('VB'):
                    verbs.append((i, token))
            
            # Look for simple SVO patterns
            for verb_idx, verb in verbs:
                # Find closest noun before verb (subject)
                subject = None
                subject_idx = -1
                for n_idx, noun in nouns:
                    if n_idx < verb_idx and (subject_idx == -1 or n_idx > subject_idx):
                        subject = noun
                        subject_idx = n_idx
                
                # Find closest noun after verb (object)
                obj = None
                obj_idx = -1
                for n_idx, noun in nouns:
                    if n_idx > verb_idx and (obj_idx == -1 or n_idx < obj_idx):
                        obj = noun
                        obj_idx = n_idx
                
                # If we found both subject and object, add the relation
                if subject and obj:
                    # Skip if either is a pronoun
                    if not any(pronoun in subject.lower() for pronoun in ["i", "you", "he", "she", "it", "we", "they"]):
                        relations.append((subject, verb, obj))
        
        return entities, relations
    
    except Exception as e:
        logger.error(f"Error in NLTK entity extraction: {str(e)}", exc_info=True)
        return [], []

def _extract_with_regex(text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    Extract entities and relations using simple regex patterns.
    This is a fallback method when NLP libraries aren't available.
    
    Args:
        text: Text to process
        
    Returns:
        Tuple of (entities, relations)
    """
    entities = []
    relations = []
    
    try:
        # Simple regex for capitalized noun phrases
        entity_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        
        # Find candidate entities
        matches = re.finditer(entity_pattern, text)
        for match in matches:
            entity_text = match.group(1)
            
            # Skip very short entities
            if len(entity_text) < 3:
                continue
            
            # Skip common words that might be capitalized
            if entity_text.lower() in ["i", "the", "a", "an", "this", "that"]:
                continue
            
            # Simple heuristic to guess entity type
            if any(term in entity_text.lower() for term in ["university", "institute", "corporation", "inc", "corp"]):
                entity_type = "ORG"
            elif any(term in entity_text.lower() for term in ["algorithm", "method", "approach", "model"]):
                entity_type = "ACADEMIC_TERM"
            else:
                entity_type = "ENTITY"  # Generic entity type
            
            entities.append((entity_text, entity_type))
        
        # Extremely simple relation extraction (not reliable)
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # Look for pattern: Entity verb entity
            words = sentence.split()
            for i in range(1, len(words)-1):
                # Check if words i-1 and i+1 are potential entities (capitalized)
                if (words[i-1][0].isupper() and 
                    not words[i][0].isupper() and 
                    words[i+1][0].isupper()):
                    
                    subject = words[i-1]
                    verb = words[i]
                    obj = words[i+1]
                    
                    # Skip common verbs and words
                    if verb.lower() not in ["is", "are", "was", "were", "a", "an", "the"]:
                        relations.append((subject, verb, obj))
        
        return entities, relations
    
    except Exception as e:
        logger.error(f"Error in regex entity extraction: {str(e)}", exc_info=True)
        return [], []

def _extract_academic_terms(text: str) -> List[Tuple[str, str]]:
    """
    Extract academic terminology from text.
    
    Args:
        text: Text to process
        
    Returns:
        List of (term, "ACADEMIC_TERM") tuples
    """
    academic_entities = []
    text_lower = text.lower()
    
    try:
        # Check for each academic term
        for term in ACADEMIC_TERMINOLOGY:
            if term in text_lower:
                # Find all occurrences (non-overlapping)
                start = 0
                while True:
                    start = text_lower.find(term, start)
                    if start == -1:
                        break
                    
                    # Get the surrounding context for better entity extraction
                    context_start = max(0, start - 20)
                    context_end = min(len(text), start + len(term) + 20)
                    context = text[context_start:context_end]
                    
                    # Use regex to extract the complete term with potential modifiers
                    # This will find terms like "deep neural network" even if we're looking for "neural network"
                    term_pattern = r'\b(\w+\s+)*' + re.escape(term) + r'(\s+\w+)*\b'
                    match = re.search(term_pattern, context, re.IGNORECASE)
                    
                    if match:
                        full_term = match.group(0).strip()
                        
                        # Skip if just a single word or already in stopwords
                        if len(full_term.split()) > 1 and not full_term.lower() in _stop_words:
                            academic_entities.append((full_term, "ACADEMIC_TERM"))
                    
                    start += len(term)
        
        return academic_entities
    
    except Exception as e:
        logger.error(f"Error extracting academic terms: {str(e)}", exc_info=True)
        return []
