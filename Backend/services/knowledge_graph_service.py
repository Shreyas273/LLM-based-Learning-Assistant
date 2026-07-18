import json
import networkx as nx
from services.llm_service import ask_llm_for_json
from services.multi_pdf_service import load_all_pdfs # Or better, query vector DB
from services.semantic_service import retrieve_context
import hashlib

# Graph Cache
graph_cache = {}

import logging

# Configure logger
logger = logging.getLogger(__name__)

def build_knowledge_graph(pdf_name=None, limit_nodes=30):
    """
    Builds a knowledge graph using Two-Stage Extraction with Production Safeguards.
    1. Extract Entities (Nouns/Keywords)
    2. Extract Relations (Entity -> Relation -> Entity)
    """
    cache_key = f"kg_{pdf_name}_{limit_nodes}"
    if cache_key in graph_cache:
        return graph_cache[cache_key]
        
    try:
        # 1. Get Text Content
        text_content = ""
        if pdf_name:
            try:
                with open(f"data/extracted_text/{pdf_name}.json", "r") as f:
                    chunks = json.load(f)
                    text_content = " ".join(chunks[:20]) 
            except Exception as e:
                logger.error(f"Failed to load PDF content for graph: {e}")
                return {"nodes": [], "edges": []}
        else:
            return {"nodes": [], "edges": []}
            
        # 2. Stage 1: Entity Extraction
        entities = extract_entities(text_content[:15000])
        logger.info(f"Extracted {len(entities)} entities for graph.")
        
        if not entities:
            return {"nodes": [], "edges": []}
            
        # 3. Stage 2: Relation Extraction
        relations = extract_relations(text_content[:15000], entities)
        logger.info(f"Extracted {len(relations)} relations for graph.")
        
        # 4. Format for Frontend
        graph_data = format_graph(entities, relations)
        
        graph_cache[cache_key] = graph_data
        return graph_data
        
    except Exception as e:
        logger.error(f"Critical Error building Knowledge Graph: {e}")
        return {"nodes": [], "edges": []}

def extract_entities(text):
    prompt = f"""
    Extract the top 20 most important technical ENTITIES (concepts, technologies, algorithms) from this text.
    - Return a Strict JSON LIST of strings.
    - Normalize: Lowercase, singularize.
    - Example: ["neural network", "transformer", "attention mechanism"]
    
    Text: {text[:4000]}...
    """
    try:
        response = ask_llm_for_json(prompt)
        if isinstance(response, list):
            return [str(e).lower() for e in response if isinstance(e, str)] # Enforce string list
        return []
    except Exception as e:
        logger.error(f"Entity Extraction Failed: {e}")
        return []

def extract_relations(text, entities):
    entities_str = ", ".join(entities)
    prompt = f"""
    Identify relationships between these entities based on the text.
    Entities: {entities_str}
    
    Return Strict JSON List of Objects:
    [
      {{"source": "entity1", "target": "entity2", "relation": "is part of"}},
      ...
    ]
    - Only use entities from the list.
    - Keep relations concise (verb phrases).
    
    Text: {text[:4000]}...
    """
    try:
        response = ask_llm_for_json(prompt)
        return response if isinstance(response, list) else []
    except Exception as e:
        logger.error(f"Relation Extraction Failed: {e}")
        return []

def format_graph(entities, relations):
    # Group entities roughly by keyword overlap or just default 'concept'
    nodes = []
    for e in entities:
        group = "concept"
        if any(x in e for x in ["server", "client", "tcp", "http"]): group = "network"
        elif any(x in e for x in ["sql", "database", "query"]): group = "database"
        elif any(x in e for x in ["algorithm", "sort", "tree"]): group = "dsa"
        
        nodes.append({
            "id": e,
            "label": e.title(),
            "group": group, # UX: Color nodes by group
            "radius": 20 # UX: Size
        })
        
    edges = []
    for r in relations:
        if r.get("source") in entities and r.get("target") in entities:
            edges.append({
                "source": r["source"],
                "target": r["target"],
                "label": r.get("relation", "relates to"),
                "id": f"{r['source']}-{r['target']}",
                "type": "curvedArrow", # UX: Arrow style
                "animated": True # UX: Animation
            })
            
    return {"nodes": nodes, "edges": edges}
