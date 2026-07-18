import json
import os
from typing import Dict, List, Tuple, Any
from datetime import datetime
from collections import Counter
import re
from services.semantic_service import get_relevant_chunks, get_confidence_score
from services.llm_service import ask_llm, ask_llm_for_json
from services.topic_extractor import extract_topics_from_chunks as extract_topics_from_chunks_clean
from services.topic_normalizer import normalize_topics
import hashlib

# Canonical list of allowed high‑level topics / technologies.
# LLM output is filtered against this list so random words
# like "Form", "Processing", "Value" never become topics.
ALLOWED_TOPICS = [
    # Core CS subjects
    "Data Structures",
    "Algorithms",
    "Design and Analysis of Algorithms",
    "Operating System",
    "Computer Organization",
    "Computer Architecture",
    "Computer Networks",
    "Database Management System",
    "Software Engineering",
    "Compiler Design",
    "Theory of Computation",
    "Discrete Mathematics",
    "Numerical Methods",
    "Distributed Systems",
    "Cloud Computing",
    "Information Security",
    "Cryptography",
    "Data Mining",
    "Data Warehousing",
    "Big Data Analytics",
    "Machine Learning",
    "Deep Learning",
    "Neural Networks",
    "Support Vector Machine",
    "Natural Language Processing",
    "Artificial Intelligence",

    # Programming paradigms
    "Object Oriented Programming",
    "Structured Programming",
    "Functional Programming",
    "Event Driven Programming",

    # Web / architecture
    "Web Development",
    "Full Stack Development",
    "Client Server Architecture",
    "REST API",
    "Microservices Architecture",
    "Design Patterns",
    "SOLID Principles",
    "Unit Testing",
    "Test Driven Development",

    # Formal languages / automata
    "Regular Expressions",
    "Context Free Grammar",
    "Finite Automata",
    "Pushdown Automata",

    # DB concepts
    "Database Normalization",
    "Relational Algebra",
    "Transaction Management",
    "Concurrency Control",

    # Networking specifics
    "TCP IP Protocol",
    "Routing Protocol",
    "Network Security",
    "Wireless Networks",

    # Languages / frameworks / stacks
    "JavaScript",
    "TypeScript",
    "Node.js",
    "React",
    "Angular",
    "Vue.js",
    "HTML",
    "CSS",
    "Bootstrap",
    "Tailwind CSS",
    "C#",
    ".NET Core",
    "ASP.NET MVC",
    "Java",
    "Spring",
    "Spring Boot",
    "Python",
    "Django",
    "Flask",
    "PHP",
    "Laravel",

    # Databases / storage
    "MongoDB",
    "SQL Server",
    "PostgreSQL",
    "MySQL",
    "Oracle Database",
    "Redis",
    "Elasticsearch",
]

def load_all_pdfs():
    """
    Load and index all uploaded PDFs for cross-referencing
    """
    pdf_index = {}
    
    if not os.path.exists("data/extracted_text"):
        return pdf_index
    
    for filename in os.listdir("data/extracted_text"):
        if filename.endswith('.json'):
            file_path = f"data/extracted_text/{filename}"
            if os.path.getsize(file_path) > 0:
                try:
                    with open(file_path, "r") as f:
                        chunks = json.load(f)
                    
                    pdf_name = os.path.splitext(filename)[0]
                    # Prefer stored metadata (produced at ingestion); fallback to clean extraction.
                    meta = {}
                    meta_path = f"data/metadata/{pdf_name}_metadata.json"
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as mf:
                                meta = json.load(mf) or {}
                        except Exception:
                            meta = {}

                    pdf_index[pdf_name] = {
                        "chunks": chunks,
                        "file_path": file_path,
                        "chunk_count": len(chunks),
                        "topics": normalize_topics(meta.get("topics") or extract_topics_from_chunks_clean(chunks)),
                        "subject": meta.get("subject"),
                        "branch": meta.get("branch"),
                        "last_modified": os.path.getmtime(file_path)
                    }
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    return pdf_index

def extract_topics_from_chunks(chunks):
    """
    Extract meaningful topics from PDF chunks.
    - Favors high‑level subject terms over noisy words.
    - Ignores common verbs / fillers like "allows", "calls", "stage", etc.
    """

    all_text = " ".join(chunks).lower()

    # ----- Stage 1: Try LLM-based canonical topic extraction -----
    try:
        allowed_str = ", ".join(ALLOWED_TOPICS)

        llm_prompt = f"""
        You are organizing a computer science syllabus.

        You are given some study material and a FIXED LIST of allowed topics:
        [{allowed_str}]

        Task:
        - Select 6-12 topics from this allowed list that BEST match the material.
        - Do NOT invent new topics or generic words.

        STRICT RULES:
        - Return ONLY a JSON list of strings chosen from the allowed list. Example:
          ["Regular Expressions", "Database Management System", "Computer Networks"]
        - Each topic must appear EXACTLY as in the allowed list.
        - NEVER return verbs or generic words like "form", "value", "processing", "primitive", "description".

        TEXT:
        {all_text[:6000]}
        """

        llm_topics = ask_llm_for_json(llm_prompt)
        if isinstance(llm_topics, list) and llm_topics:
            cleaned = []
            seen = set()
            for t in llm_topics:
                if not isinstance(t, str):
                    continue
                name = t.strip()
                if not name:
                    continue
                # Enforce membership in allowed topics
                if name not in ALLOWED_TOPICS:
                    continue
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(name)

            # If LLM produced a reasonable topic list, use it directly.
            if cleaned:
                return cleaned[:15]
    except Exception as e:
        # Fall back silently to heuristic extraction below if LLM fails.
        print(f"LLM topic extraction failed, using heuristic: {e}")

    # ----- Stage 2: Heuristic keyword-based topics (fallback) -----
    # Broad stopword list: standard fillers + domain‑specific verbs / noise
    stopwords = {
        # generic english / glue words
        "the","and","for","with","from","this","that","these","those","such",
        "into","onto","about","above","below","under","over","between","within",
        "also","very","much","many","other","some","any","each","every",
        "can","could","may","might","must","should","would","will","shall",
        "is","are","was","were","be","being","been","has","have","had","do",
        "does","did","done","using","used","use","based","given","make",
        "makes","made","provide","provides","provided","show","shows","shown",
        "allow","allows","allowed","call","calls","called","stage","stages",
        "type","types","various","different","example","examples","system",
        "section","chapter","figure","table",
        # programming / low‑signal tokens
        "class","ip","process","var","int","string","object","function",
        "data","information","value","values","result","results","method",
        "methods","field","fields","parameter","parameters","argument",
        # misc
        "etc","eg","ie"
    }

    # Detect important predefined multi‑word topics first
    # This list should cover most core CS subjects and common stacks.
    topic_candidates = [
        # Core CS subjects
        "data structures",
        "algorithms",
        "design and analysis of algorithms",
        "operating system",
        "computer organization",
        "computer architecture",
        "computer networks",
        "database management system",
        "software engineering",
        "compiler design",
        "theory of computation",
        "discrete mathematics",
        "numerical methods",
        "distributed systems",
        "cloud computing",
        "information security",
        "cryptography",
        "data mining",
        "data warehousing",
        "big data analytics",
        "machine learning",
        "deep learning",
        "neural networks",
        "support vector machine",
        "natural language processing",
        "artificial intelligence",

        # Web and programming
        "object oriented programming",
        "structured programming",
        "functional programming",
        "event driven programming",
        "web development",
        "full stack development",
        "client server architecture",
        "rest api",
        "microservices architecture",
        "design patterns",
        "solid principles",
        "unit testing",
        "test driven development",

        # Regular expressions / formal languages
        "regular expressions",
        "context free grammar",
        "finite automata",
        "pushdown automata",

        # Database specifics
        "database normalization",
        "relational algebra",
        "transaction management",
        "concurrency control",

        # Networking specifics
        "tcp ip protocol",
        "routing protocol",
        "network security",
        "wireless networks",

        # Technology / stack oriented (languages, frameworks, DBs)
        "javascript",
        "type script",
        "node.js",
        "react js",
        "angular js",
        "vue js",
        "html",
        "cascading style sheets",
        "css",
        "bootstrap framework",
        "tailwind css",
        "asp.net mvc",
        "dotnet core",
        "c sharp",
        "java programming",
        "spring framework",
        "spring boot",
        "python programming",
        "django framework",
        "flask framework",
        "php programming",
        "laravel framework",

        # Databases / storage
        "mongo db",
        "sql server",
        "postgresql",
        "mysql database",
        "oracle database",
        "redis cache",
        "elasticsearch",
    ]

    topics = []

    for topic in topic_candidates:
        if topic in all_text:
            # Normalize some common technology phrases to cleaner topic names
            pretty = topic.title()
            if topic == "javascript":
                pretty = "JavaScript"
            elif topic == "type script":
                pretty = "TypeScript"
            elif topic == "node.js":
                pretty = "Node.js"
            elif topic == "react js":
                pretty = "React"
            elif topic == "asp.net mvc":
                pretty = "ASP.NET MVC"
            elif topic == "dotnet core":
                pretty = ".NET Core"
            elif topic == "c sharp":
                pretty = "C#"
            elif topic == "mongo db":
                pretty = "MongoDB"
            elif topic == "mysql database":
                pretty = "MySQL"

            topics.append(pretty)

    # Fallback: extract frequent, content‑heavy keywords from the text
    # Tokenize to words (alphabetic + digits, at least 4 chars)
    tokens = re.findall(r"[a-z][a-z0-9_-]{3,}", all_text)

    # Filter out stopwords / obvious noise
    content_tokens = [
        t
        for t in tokens
        if t not in stopwords
        and not t.isnumeric()
        and len(t) >= 4
    ]

    if not content_tokens:
        return list(set(topics))[:12]

    freq = Counter(content_tokens)

    # Only keep words that appear multiple times to avoid random one‑offs
    for word, count in freq.most_common(80):
        if count < 3:
            # below this threshold terms tend to be noise
            break
        topics.append(word.replace("_", " "))

    # De‑duplicate while preserving order, limit to 12–15 main topics
    seen = set()
    unique_topics = []
    for t in topics:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique_topics.append(t.title())

    return unique_topics[:15]

def cross_reference_search(query, top_k=3):
    """
    Search across all PDFs using Vector DB with Source Ranking.
    Strategies:
    1. Fetch wide pool of chunks (top_k * 10).
    2. Group by Source PDF.
    3. Score Sources: (MaxChunkScore * 0.7) + (AvgChunkScore * 0.3).
    4. Return Top Sources with Top Chunks.
    """
    from services.semantic_service import retrieve_context

    def _is_comparison_question(q: str) -> bool:
        if not q:
            return False
        s = q.lower()
        triggers = [
            "compare", "comparison", "differentiate", "difference between", "differences between",
            "vs", "versus", "pros and cons", "advantages and disadvantages",
        ]
        return any(t in s for t in triggers)
    
    # 1. Fetch wide pool
    # Multi-hop retrieval for comparison questions: break into sub-queries and merge.
    subqueries = [query]
    if _is_comparison_question(query):
        try:
            hop_prompt = f"""
You break comparison questions into 2-4 focused retrieval sub-queries.
Return STRICT JSON only:
{{
  "queries": ["subquery 1", "subquery 2"]
}}

Question: {query}
"""
            hop = ask_llm_for_json(hop_prompt)
            if isinstance(hop, dict) and isinstance(hop.get("queries"), list) and hop["queries"]:
                subqueries = [str(x).strip() for x in hop["queries"] if str(x).strip()][:4]
        except Exception:
            subqueries = [query]

    merged_results = []
    seen = set()
    for sq in subqueries:
        results_sq = retrieve_context(sq, top_k=30)
        for item in results_sq or []:
            meta = item.get("metadata") or {}
            key = (
                str(meta.get("pdf_name") or meta.get("source") or ""),
                str(meta.get("page_number") or meta.get("page") or ""),
                str(meta.get("chunk_id") or ""),
                hash((item.get("text") or "").strip()),
            )
            if key in seen:
                continue
            seen.add(key)
            merged_results.append(item)

    results = sorted(merged_results, key=lambda x: float(x.get("score") or 0.0), reverse=True)
    
    if not results:
        return []
        
    # 2. Group by Source
    sources = {}
    for item in results:
        meta = item.get("metadata") or {}
        source = meta.get("pdf_name") or meta.get("source") or "unknown"
        if source not in sources:
            sources[source] = {
                "chunks": [],
                "scores": []
            }
        sources[source]["chunks"].append(item)
        sources[source]["scores"].append(item['score'])
        
    # 3. Rank Sources
    ranked_sources = []
    for source, data in sources.items():
        max_score = max(data["scores"])
        avg_score = sum(data["scores"]) / len(data["scores"])
        
        # Heuristic Ranking Score
        rank_score = (max_score * 0.7) + (avg_score * 0.3)
        
        if rank_score > 40: # Filter low relevance sources
            ranked_sources.append({
                "source": source,
                "rank_score": rank_score,
                "top_chunks": sorted(data["chunks"], key=lambda x: x['score'], reverse=True)[:2] # Top 2 chunks
            })
            
    # Sort by rank
    ranked_sources.sort(key=lambda x: x["rank_score"], reverse=True)
    
    # Format for downstream
    final_output = []
    for item in ranked_sources[:top_k]:
        final_output.append({
            "pdf_name": item["source"],
            "chunks": [c['text'] for c in item["top_chunks"]],
            "average_relevance": item["rank_score"],
            "chunk_count": len(item["top_chunks"]),
            "top_sources": [
                {
                    "pdf": (c.get("metadata") or {}).get("pdf_name") or (c.get("metadata") or {}).get("source") or item["source"],
                    "page": (c.get("metadata") or {}).get("page_number", (c.get("metadata") or {}).get("page", None)),
                    "relevance_score": float(c.get("score") or 0.0),
                }
                for c in item["top_chunks"]
            ],
        })
        
    return final_output

def generate_cross_referenced_answer(question, pdf_index=None, difficulty="medium"):
    """
    Generate answer using information from multiple PDFs via Vector DB.
    """
    try:
        # 1. Search with new Ranker
        search_results = cross_reference_search(question, top_k=3)
        
        if not search_results:
             return "No relevant cross-reference information found.", 0.0, []

        # 2. Format Context for LLM
        context_str = ""
        sources_list = []
        citation_counter = 0
        
        for res in search_results:
            source_name = res["pdf_name"]
            relevance = res["average_relevance"]
            chunks = res["chunks"]
            
            context_str += f"\nSOURCE: {source_name} (Relevance: {relevance:.0f}%)\n"
            for i, chunk in enumerate(chunks):
                citation_counter += 1
                # Use the same citation convention as llm_service: [S#]
                top_meta = None
                try:
                    top_meta = (res.get("top_sources") or [None])[i]
                except Exception:
                    top_meta = None
                pdf = (top_meta or {}).get("pdf") or source_name
                page = (top_meta or {}).get("page", None)
                context_str += f"- [S{citation_counter} | pdf={pdf} | page={page}] {chunk}\n"
            
            # Return normalized sources while keeping backward-compatible keys.
            for s in res.get("top_sources", []) or []:
                pdf = s.get("pdf") or source_name
                page = s.get("page", None)
                try:
                    page = None if page is None else int(page)
                except Exception:
                    page = None
                sources_list.append({
                    "citation_id": f"S{len(sources_list) + 1}",
                    "pdf": pdf,
                    "page": page,
                    "relevance_score": float(s.get("relevance_score") or relevance or 0.0),
                    # legacy-friendly aliases
                    "pdf_name": pdf,
                    "source": pdf,
                    "relevance": float(s.get("relevance_score") or relevance or 0.0),
                    "topics": [],
                })
            
        # 3. Call LLM with "cross_reference" mode
        answer, confidence, _, _ = ask_llm(
            question, 
            relevant_context=[context_str], # Pass pre-formatted string list
            difficulty=difficulty, 
            enable_fact_check=False,
            study_approach="cross_reference" # Triggers table output
        )
        
        return answer, confidence, sources_list
        
    except Exception as e:
        return f"Error generating cross-referenced answer: {str(e)}", 0.0, []

def find_related_topics(current_topic, pdf_index):
    """
    Find topics related to current topic across all PDFs
    """
    if not pdf_index:
        return []
    
    related_topics = {}
    
    for pdf_name, pdf_data in pdf_index.items():
        pdf_topics = pdf_data["topics"]
        
        # Find topics that appear with current topic
        if current_topic.lower() in " ".join(pdf_topics).lower():
            for topic in pdf_topics:
                if topic.lower() != current_topic.lower():
                    if topic not in related_topics:
                        related_topics[topic] = []
                    related_topics[topic].append(pdf_name)
    
    # Sort by frequency (most related topics first)
    sorted_topics = sorted(related_topics.items(), key=lambda x: len(x[1]), reverse=True)
    
    return [{"topic": topic, "pdfs": pdfs} for topic, pdfs in sorted_topics[:5]]

def generate_comparative_analysis(topic, pdf_index=None):
    """
    Generate comparative analysis of a topic across multiple PDFs using Vector DB
    """
    from services.semantic_service import retrieve_context
    
    # Retrieve relevant chunks for the topic
    results = retrieve_context(topic, top_k=10)
    
    if not results:
        return f"No information found about '{topic}' in uploaded PDFs.", []
        
    topic_coverage = {}
    # Group by PDF
    for item in results:
        pdf_name = item['metadata']['source']
        if pdf_name not in topic_coverage:
             topic_coverage[pdf_name] = {
                "relevance_score": 0,
                "relevant_chunks": [],
                "coverage_percentage": 0 # heuristic
             }
        topic_coverage[pdf_name]["relevant_chunks"].append(item['text'])
        topic_coverage[pdf_name]["relevance_score"] += item['score']

    # Calculate coverage percentage heuristic
    for pdf in topic_coverage:
        # avg score?
        score = topic_coverage[pdf]["relevance_score"] / len(topic_coverage[pdf]["relevant_chunks"])
        topic_coverage[pdf]["coverage_percentage"] = score

    # Generate comparative analysis
    analysis = f"**Comparative Analysis: {topic}**\n\n"
    
    for pdf_name, data in topic_coverage.items():
        analysis += f"**{pdf_name}** (Coverage: {data['coverage_percentage']:.1f}%)\n"
        
        if data["relevant_chunks"]:
            # Summarize key points from this PDF
            combined_text = " ".join(data["relevant_chunks"])
            summary = combined_text[:200] + "..." if len(combined_text) > 200 else combined_text
            analysis += f"Key points: {summary}\n\n"
    
    return analysis, list(topic_coverage.keys())

def create_knowledge_graph(pdf_index):
    """
    Knowledge graph structure:
      subject → topic → subtopic

    Nodes:
      { id, label, type: "subject" | "topic" | "subtopic" }
    Edges:
      subject → topic
      topic → subtopic
    """

    def _safe_id(prefix: str, label: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(label or "").strip())
        return f"{prefix}_{s}".strip("_")

    def _build_hierarchy(topics: List[str]) -> Dict[str, List[str]]:
        """
        Heuristic topic→subtopic mapping (DSA-friendly).
        If a topic looks "specific", map it under a broader parent if possible.
        """
        parents = {}
        for t in topics:
            parents[t] = []

        # Common parent buckets
        parent_keywords = {
            "Tree": ["tree", "bst", "avl", "red black", "b-tree", "binary search tree", "heap"],
            "Graph": ["graph", "bfs", "dfs", "dijkstra", "bellman", "mst", "kruskal", "prim"],
            "Sorting": ["sort", "merge sort", "quick sort", "heap sort", "counting sort", "radix"],
            "Searching": ["search", "binary search", "linear search"],
            "Dynamic Programming": ["dynamic programming", "dp", "memoization", "tabulation"],
        }

        # Ensure parent topics exist if there are matches
        norm = {t.lower(): t for t in topics}
        for parent in list(parent_keywords.keys()):
            if parent.lower() not in norm and any(any(k in tl for k in ks) for tl in norm.keys() for ks in [parent_keywords[parent]]):
                topics.append(parent)
                norm[parent.lower()] = parent
                parents[parent] = []

        # Assign specific topics as subtopics if they contain parent keywords and aren't identical
        for t in list(parents.keys()):
            tl = t.lower()
            for parent, kws in parent_keywords.items():
                if parent.lower() == tl:
                    continue
                if any(k in tl for k in kws):
                    parents.setdefault(parent, [])
                    if t not in parents[parent]:
                        parents[parent].append(t)
                    break

        # Remove self-mappings and duplicates
        for p in list(parents.keys()):
            uniq = []
            seen = set()
            for s in parents[p]:
                if s == p:
                    continue
                k = s.lower()
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(s)
            parents[p] = uniq

        return parents

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    node_seen = set()

    for pdf_name, data in (pdf_index or {}).items():
        subject_label = data.get("subject") or data.get("branch") or "Unknown Subject"
        subj_id = _safe_id("subject", subject_label)
        if subj_id not in node_seen:
            nodes.append({"id": subj_id, "label": subject_label, "type": "subject"})
            node_seen.add(subj_id)

        topics = normalize_topics(data.get("topics") or [])
        hierarchy = _build_hierarchy(list(topics))

        # Create topic nodes + edges subject→topic
        for topic in hierarchy.keys():
            topic_id = _safe_id("topic", topic)
            if topic_id not in node_seen:
                nodes.append({"id": topic_id, "label": topic, "type": "topic"})
                node_seen.add(topic_id)
            edges.append({"source": subj_id, "target": topic_id})

            # Create subtopic nodes + edges topic→subtopic
            for sub in hierarchy.get(topic, []) or []:
                sub_id = _safe_id("subtopic", sub)
                if sub_id not in node_seen:
                    nodes.append({"id": sub_id, "label": sub, "type": "subtopic"})
                    node_seen.add(sub_id)
                edges.append({"source": topic_id, "target": sub_id})

    return {"nodes": nodes, "edges": edges}
    
def recommend_additional_resources(current_question, pdf_index, user_history=None):
    """
    Recommend additional PDFs or topics based on current question and user history
    """
    if not pdf_index:
        return []
    
    recommendations = []
    
    # Find PDFs that might help with current question
    cross_ref_results = cross_reference_search(current_question, top_k=5)
    
    for result in cross_ref_results:
        pdf_name = result["pdf_name"]
        
        # Check if user has interacted with this PDF recently
        recently_used = False
        if user_history:
            for item in user_history[-10:]:  # Check last 10 interactions
                if pdf_name.lower() in item.get("answer", "").lower():
                    recently_used = True
                    break
        
        if not recently_used:
            recommendations.append({
                "type": "pdf",
                "name": pdf_name,
                "reason": f"Contains relevant information (Relevance: {result['average_relevance']:.1f}%)",
                "topics": []
            })
    
    # Find related topics
    current_topic = extract_main_topic(current_question)
    if current_topic:
        related_topics = find_related_topics(current_topic, pdf_index)
        
        for topic_data in related_topics[:3]:
            recommendations.append({
                "type": "topic",
                "name": topic_data["topic"],
                "reason": f"Related to {current_topic}",
                "pdfs": topic_data["pdfs"]
            })
    
    return recommendations[:5]  # Limit recommendations

def extract_main_topic(question):
    """
    Extract the main topic from a question
    """
    question_lower = question.lower()
    
    topic_keywords = {
        "algorithm": ["algorithm", "sort", "search", "complexity"],
        "data structure": ["data structure", "array", "list", "tree", "graph"],
        "operating system": ["operating system", "process", "thread", "memory", "scheduling"],
        "database": ["database", "sql", "query", "table", "index"],
        "network": ["network", "protocol", "tcp", "ip", "routing"],
        "programming": ["programming", "code", "function", "class", "object"]
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            return topic
    
    return "general"

def update_pdf_metadata(pdf_name, new_topics=None):
    """
    Update metadata for a specific PDF
    """
    try:
        file_path = f"data/extracted_text/{pdf_name}.json"
        if not os.path.exists(file_path):
            return False
        
        # Load existing data
        with open(file_path, "r") as f:
            chunks = json.load(f)
        
        # Update topics if provided
        if new_topics:
            topics = new_topics
        else:
            topics = extract_topics_from_chunks(chunks)
        
        # Create metadata file
        metadata = {
            "pdf_name": pdf_name,
            "chunk_count": len(chunks),
            "topics": topics,
            "last_updated": datetime.now().isoformat(),
            "file_hash": calculate_file_hash(file_path)
        }
        
        metadata_path = f"data/metadata/{pdf_name}_metadata.json"
        os.makedirs("data/metadata", exist_ok=True)
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error updating metadata for {pdf_name}: {e}")
        return False

def calculate_file_hash(file_path):
    """
    Calculate hash of file for integrity checking
    """
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()
    except:
        return ""
