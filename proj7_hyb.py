import os
import time
import psutil
import tracemalloc
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader
import chromadb
from neo4j import GraphDatabase
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')
load_dotenv()

# Initialize models and databases
class RAGSystems:
    def __init__(self):
        st.write("Initializing RAG systems...")
        self.memory_usage = {
            "naive_rag": {"retrieval": [], "generation": []},
            "graph_rag": {"retrieval": [], "generation": []},
            "hybrid_rag": {"retrieval": [], "generation": []}
        } 
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            st.write("Embedding model loaded successfully")
        except Exception as e:
            st.error(f"Failed to load embedding model: {str(e)}")
            raise
            
        # Initialize ChromaDB for naive RAG
        try:
            self.chroma_client = chromadb.PersistentClient(path="chroma_db")
            st.write("ChromaDB initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
            
        # Initialize Neo4j for graph RAG
        try:
            self.neo4j_driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                auth=(os.getenv("NEO4J_USER", "neo4j"), 
                      os.getenv("NEO4J_PASSWORD", "password")))
            st.write("Neo4j driver initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize Neo4j driver: {str(e)}")
            st.warning("Graph RAG features will be disabled")
            self.neo4j_driver = None
            
        # Initialize Gemini
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.gemini = genai.GenerativeModel('models/gemini-1.5-pro-latest')
            st.write("Gemini model loaded successfully")
        except Exception as e:
            st.error(f"Failed to initialize Gemini: {str(e)}")
            raise
        self.vector_storage_path = "vector_storage"
        os.makedirs(self.vector_storage_path, exist_ok=True)
    
    # Add vector file tracking
        self.vector_files = {
            "naive_rag": os.path.join(self.vector_storage_path, "naive_rag_vectors.npy"),
            "naive_rag_ids": os.path.join(self.vector_storage_path, "naive_rag_ids.npy")
        }    
        # Create collections/graph if they don't exist
        self._initialize_databases()
    
    def _initialize_databases(self):
        # ChromaDB collection
        try:
            self.chroma_collection = self.chroma_client.get_collection("naive_rag")
            st.write("Connected to existing ChromaDB collection")
        except:
            try:
                self.chroma_collection = self.chroma_client.create_collection("naive_rag")
                st.write("Created new ChromaDB collection")
            except Exception as e:
                st.error(f"Failed to create ChromaDB collection: {str(e)}")
                raise
        
        # Neo4j schema (only if Neo4j is available)
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE
                    """)
                    session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (n:Document) REQUIRE n.name IS UNIQUE
                    """)
                st.write("Neo4j constraints created successfully")
            except Exception as e:
                st.error(f"Failed to create Neo4j constraints: {str(e)}")
                st.warning("Graph RAG features may not work properly")
    
    def process_documents(self, files, clear_existing=True):
        """Process uploaded documents and store in all RAG systems"""
        if not files:
            st.warning("No files provided")
            return
        if clear_existing:
            self.clear_chroma_collection()  # Clear existing data before processing new files
            if hasattr(self, 'light_rag_data'):
                self.light_rag_data = {"chunks": [], "embeddings": [], "document_map": {}}
        
        st.write(f"Processing {len(files)} documents...")
        documents = []
        
        for file in files:
            try:
                if file.name.endswith('.pdf'):
                    text = self._extract_text_from_pdf(file)
                else:
                    text = file.getvalue().decode('utf-8')
                documents.append((file.name, text))
                st.write(f"Processed {file.name}")
            except Exception as e:
                st.error(f"Failed to process {file.name}: {str(e)}")
                continue
        
        if not documents:
            st.error("No documents were successfully processed")
            return
        
        # Process for naive RAG (ChromaDB)
        try:
            self._process_for_naive_rag(documents)
            st.success("Documents processed for Naive RAG")
        except Exception as e:
            st.error(f"Failed to process for Naive RAG: {str(e)}")
        
        # Process for graph RAG (Neo4j)
        if self.neo4j_driver:
            try:
                self._process_for_graph_rag(documents)
                st.success("Documents processed for Graph RAG")
            except Exception as e:
                st.error(f"Failed to process for Graph RAG: {str(e)}")
        else:
            st.warning("Skipping Graph RAG processing (Neo4j not available)")
        
        # Light RAG doesn't need preprocessing
        try:
            self._process_for_hybrid_rag(documents)
            st.success("Documents processed for Hybrid RAG")
        except Exception as e:
            st.error(f"Failed to process for Hybrid RAG: {str(e)}")
    
    def _extract_text_from_pdf(self, pdf_file):
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"PDF extraction error: {str(e)}")
            raise
    
    def _chunk_text(self, text, chunk_size=500):
        """Simple text chunking"""
        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
    
    def _process_for_naive_rag(self, documents):
        """Store documents in ChromaDB with embeddings"""
        ids = []
        documents_list = []
        embeddings = []
        
        for idx, (name, text) in enumerate(documents):
            chunks = self._chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{name}_{chunk_idx}"
                embedding = self.embedding_model.encode(chunk)
                
                ids.append(chunk_id)
                documents_list.append(chunk)
                embeddings.append(embedding.tolist())
        
        # Store in ChromaDB
        self.chroma_collection.add(
            ids=ids,
            documents=documents_list,
            embeddings=embeddings
        )
        self._save_vectors_to_file(ids, np.array(embeddings), "naive_rag")
    def _save_vectors_to_file(self, ids, embeddings, system_name):
        """Save vectors and IDs to numpy files"""
        try:
            # Save embeddings
            np.save(self.vector_files[system_name], embeddings)
        
        # Save IDs
            np.save(self.vector_files[f"{system_name}_ids"], np.array(ids))
        
            st.write(f"Vectors saved to file for {system_name}")
        except Exception as e:
            st.error(f"Failed to save vectors to file: {str(e)}")

    def _load_vectors_from_file(self, system_name):
        """Load vectors and IDs from numpy files"""
        try:
            embeddings = np.load(self.vector_files[system_name])
            ids = np.load(self.vector_files[f"{system_name}_ids"])
            return ids, embeddings
        except Exception as e:
            st.error(f"Failed to load vectors from file: {str(e)}")
            return None, None
    def display_vector_by_id(self, chunk_id):
        """Display the vector for a specific chunk ID with proper error handling"""
        try:
            if not chunk_id or not isinstance(chunk_id, str):
                st.warning("Please enter a valid chunk ID")
                return

        # Try ChromaDB first
            try:
                result = self.chroma_collection.get(ids=[chunk_id], include=["embeddings"])
                if result and 'embeddings' in result and result['embeddings']:
                    vector = np.array(result['embeddings'][0])
                    st.write(f"📊 Vector for '{chunk_id}' (from ChromaDB)")
                    st.code(f"Shape: {vector.shape}\n{vector}")
                    return
            except Exception as db_error:
                st.warning(f"ChromaDB lookup warning: {str(db_error)}")

        # Fall back to file storage
            try:
                ids, embeddings = self._load_vectors_from_file("naive_rag")
                if ids is not None and embeddings is not None:
                # Convert both to numpy arrays if they aren't already
                    ids_array = np.array(ids)
                    embeddings_array = np.array(embeddings)
                
                # Find matching indices safely
                    matching_indices = np.flatnonzero(ids_array == chunk_id)
                
                    if matching_indices.size > 0:
                        vector = embeddings_array[matching_indices[0]]
                        st.write(f"📊 Vector for '{chunk_id}' (from file backup)")
                        st.code(f"Shape: {vector.shape}\n{vector}")
                        return
            except Exception as file_error:
                st.warning(f"File lookup warning: {str(file_error)}")

            st.error(f"Vector not found for ID: '{chunk_id}'")
        
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

    def display_random_vectors(self, n=3):
        """Display sample vectors with bulletproof error handling"""
        try:
            st.subheader("🔍 Sample Vectors from Naive RAG")
        
        # Try ChromaDB first
            chroma_success = False
            try:
                chroma_ids = self.chroma_collection.get()['ids']
                if len(chroma_ids) > 0:
                    selected_ids = np.random.choice(chroma_ids, min(n, len(chroma_ids)), replace=False)
                    for chunk_id in selected_ids:
                        self.display_vector_by_id(chunk_id)
                        st.write("---")
                    chroma_success = True
            except Exception as e:
                st.warning(f"ChromaDB sampling issue: {str(e)}")

        # If ChromaDB failed, try file storage
            if not chroma_success:
                try:
                    ids, embeddings = self._load_vectors_from_file("naive_rag")
                    if ids is not None and len(ids) > 0:
                        selected_indices = np.random.choice(len(ids), min(n, len(ids)), replace=False)
                        for idx in selected_indices:
                            st.write(f"📊 Random Vector {idx+1}/{n}")
                            st.code(f"ID: {ids[idx]}\nShape: {embeddings[idx].shape}\n{embeddings[idx]}")
                            st.write("---")
                        return
                except Exception as e:
                    st.error(f"File sampling failed: {str(e)}")

            st.warning("No vectors available in any storage backend")

        except Exception as e:
            st.error(f"Critical error in random sampling: {str(e)}")

    def display_vector_stats(self):
        """Show statistics about the stored vectors"""
        try:
            stats = {}
        
            # Try ChromaDB
            try:
                chroma_count = self.chroma_collection.count()
                stats["ChromaDB Vectors Count"] = chroma_count
            except:
                stats["ChromaDB Status"] = "Not available"
        
            # Check file storage
            try:
                ids, embeddings = self._load_vectors_from_file("naive_rag")
                if embeddings is not None:
                    stats["File Storage Vectors Count"] = len(embeddings)
                    stats["Vector Dimensions"] = embeddings.shape[1] if len(embeddings) > 0 else 0
                    stats["Mean Vector Magnitude"] = np.mean(np.linalg.norm(embeddings, axis=1)) if len(embeddings) > 0 else 0
            except:
                stats["File Storage Status"] = "Not available"
        
            # Display stats
            st.subheader("Vector Statistics")
            for k, v in stats.items():
                st.write(f"{k}: {v}")
        
            # Show sample vectors if available
            self.display_random_vectors(2)
    
        except Exception as e:
            st.error(f"Error calculating vector stats: {str(e)}")
    def _process_for_graph_rag(self, documents):
        """Store documents in Neo4j with relationships"""
        with self.neo4j_driver.session() as session:
            for name, text in documents:
                # Create document node
                session.run("""
                MERGE (d:Document {name: $name})
                """, name=name)
                
                # Split into chunks and create relationships
                chunks = self._chunk_text(text)
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{name}_{chunk_idx}"
                    embedding = self.embedding_model.encode(chunk).tolist()
                    
                    session.run("""
                    MERGE (c:Chunk {id: $id, text: $text, embedding: $embedding})
                    MERGE (d:Document {name: $name})
                    MERGE (d)-[:CONTAINS]->(c)
                    """, id=chunk_id, text=chunk, embedding=embedding, name=name)
                    
                    # Link similar chunks (simplified)
                    if chunk_idx > 0:
                        prev_id = f"{name}_{chunk_idx-1}"
                        session.run("""
                        MATCH (c1:Chunk {id: $id1}), (c2:Chunk {id: $id2})
                        MERGE (c1)-[:NEXT]->(c2)
                        """, id1=prev_id, id2=chunk_id)
    
    def query_naive_rag(self, question, k=3, use_file_backup=False):
        """Query naive RAG with option to use file backup"""
        try:
            question_embedding = self.embedding_model.encode(question).tolist()
        
            if use_file_backup:
                # Load from file backup
                ids, embeddings = self._load_vectors_from_file("naive_rag")
                if embeddings is None:
                    raise Exception("Could not load vectors from file")
                
            # Calculate similarities manually
                similarities = cosine_similarity(
                    [question_embedding],
                    embeddings
                )[0]
            
            # Get top k results
                top_k_indices = np.argsort(similarities)[::-1][:k]
                top_ids = ids[top_k_indices]
                top_documents = []
            
            # Get documents from ChromaDB by ID
                results = self.chroma_collection.get(ids=top_ids.tolist())
                context = "\n\n".join(results['documents'])
            else:
            # Normal ChromaDB query
                results = self.chroma_collection.query(
                    query_embeddings=[question_embedding],
                    n_results=k
                )
                context = "\n\n".join(results['documents'][0])
        
            prompt = f"""Use the following context to answer the question:
        
            Context:
            {context}
        
            Question: {question}
        
            Answer:"""
        
            response = self.gemini.generate_content(prompt)
            return response.text, context
        except Exception as e:
            st.error(f"Naive RAG query failed: {str(e)}")
            return f"Error: {str(e)}", ""
    def clear_chroma_collection(self):
        """Completely clear the ChromaDB collection"""
        try:
            self.chroma_client.delete_collection("naive_rag")
            self.chroma_collection = self.chroma_client.create_collection("naive_rag")
            st.success("ChromaDB collection cleared successfully")
        except Exception as e:
            st.error(f"Error clearing ChromaDB: {str(e)}")
    def query_graph_rag(self, question, k=3):
        """Query graph RAG (graph traversal + vector similarity in Python)"""
        if not self.neo4j_driver:
            return "Graph RAG not available (Neo4j not configured)", ""

        try:
            question_embedding = self.embedding_model.encode(question).tolist()

            with self.neo4j_driver.session() as session:
                result = session.run("""
                MATCH (c:Chunk)
                RETURN c.text AS text, c.embedding AS embedding
                """)

            # Get all nodes and their embeddings
                chunks = []
                for record in result:
                    if record["embedding"]:  # check for nulls
                        chunks.append({
                            "text": record["text"],
                            "embedding": record["embedding"]
                        })

        # Compute cosine similarity in Python
            embeddings_matrix = np.array([chunk["embedding"] for chunk in chunks])
            similarities = cosine_similarity(
                [question_embedding], embeddings_matrix
            )[0]  # shape: (n_chunks,)

        # Sort and get top k
            top_k_indices = np.argsort(similarities)[::-1][:k]
            top_chunks = [chunks[i] for i in top_k_indices]
            context = "\n\n".join([chunk["text"] for chunk in top_chunks])

        # Prompt for Gemini
            prompt = f"""Use the following context to answer the question:

            Context:
            {context}

            Question: {question}

            Answer:"""

            response = self.gemini.generate_content(prompt)
            return response.text, context

        except Exception as e:
            st.error(f"Graph RAG query failed: {str(e)}")
            return f"Error: {str(e)}", ""
        
    def _process_for_hybrid_rag(self, documents):
        """Store documents in both ChromaDB and Neo4j with cross-references"""
        # First process for naive RAG (ChromaDB)
        self._process_for_naive_rag(documents)
        
        # Then process for graph RAG (Neo4j) with additional relationships
        if self.neo4j_driver:
            with self.neo4j_driver.session() as session:
                for name, text in documents:
                    # Create document node with vector store reference
                    session.run("""
                    MERGE (d:Document {name: $name})
                    SET d.vector_store = 'chroma'
                    """, name=name)
                    
                    # Create chunks with cross-store references
                    chunks = self._chunk_text(text)
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_id = f"{name}_{chunk_idx}"
                        embedding = self.embedding_model.encode(chunk).tolist()
                        
                        # Create chunk node with dual storage references
                        session.run("""
                        MERGE (c:Chunk {id: $id})
                        SET c.text = $text,
                            c.embedding = $embedding,
                            c.vector_id = $id  
                        MERGE (d:Document {name: $name})
                        MERGE (d)-[:CONTAINS]->(c)
                        """, id=chunk_id, text=chunk, 
                                  embedding=embedding, name=name)
    def query_hybrid_rag(self, question, k=3):
        """Hybrid RAG query combining vector and graph search"""
        try:
            # Phase 1: Vector search from ChromaDB
            question_embedding = self.embedding_model.encode(question).tolist()
            chroma_results = self.chroma_collection.query(
                query_embeddings=[question_embedding],
                n_results=k
            )
            top_ids = chroma_results['ids'][0]
            
            # Phase 2: Graph expansion in Neo4j
            if self.neo4j_driver:
                with self.neo4j_driver.session() as session:
                    result = session.run("""
                    UNWIND $ids AS vector_id
                    MATCH (c:Chunk {vector_id: vector_id})-[:NEXT*0..2]-(related)
                    RETURN DISTINCT related.text AS text
                    LIMIT $double_k
                    """, ids=top_ids, double_k=k*2)
                    
                    graph_chunks = [record["text"] for record in result]
            else:
                graph_chunks = []
            
            # Combine results
            chroma_context = "\n\n".join(chroma_results['documents'][0])
            graph_context = "\n\n".join(graph_chunks)
            combined_context = f"Vector Results:\n{chroma_context}\n\nGraph Results:\n{graph_context}"
            
            # Generate answer
            prompt = f"""Combine information from both vector and graph results:
            
            {combined_context}
            
            Question: {question}
            
            Answer:"""
            
            response = self.gemini.generate_content(prompt)
            return response.text, combined_context
        
        except Exception as e:
            st.error(f"Hybrid RAG query failed: {str(e)}")
            return f"Error: {str(e)}", ""

    
    def evaluate_performance(self, question, ground_truth, k=3):
        """Evaluate all RAG systems with comprehensive metrics"""
        results = {}
        tracemalloc.start()
        
        # Query all systems with detailed measurements
        naive_result = self._evaluate_naive_rag(question, ground_truth, k)
        graph_result = self._evaluate_graph_rag(question, ground_truth, k)
        hybrid_result = self._evaluate_hybrid_rag(question, ground_truth, k)
        
        # Compile results
        results = {
            "Naive RAG": naive_result,
            "Graph RAG": graph_result,
            "Hybrid RAG": hybrid_result
        }
        
        tracemalloc.stop()
        return results
    
    def _evaluate_naive_rag(self, question, ground_truth, k=3):
        """Evaluate Naive RAG with detailed metrics including file backup option"""
        result = {
            "answer": "",
            "context": "",
            "metrics": {
                "used_backup": False,
                "precision@k": 0.0,
                "recall@k": 0.0,
                "f1_score": 0.0,
                "bleu_score": 0.0,
                "retrieval_latency_ms": 0.0,
                "generation_latency_ms": 0.0,
                "end_to_end_latency_ms": 0.0,
                "retrieval_memory_mb": 0.0,
                "generation_memory_mb": 0.0,
                "hallucination_rate": 0.0
            }
        }
    
        try:
            # Measure retrieval - try primary method first
            retrieval_start_time = time.time()
            retrieval_start_mem = psutil.virtual_memory().used / (1024 * 1024)  # MB
        
            question_embedding = self.embedding_model.encode(question).tolist()
        
            try:
                # Primary ChromaDB retrieval
                chroma_results = self.chroma_collection.query(
                    query_embeddings=[question_embedding],
                    n_results=k
                )
                retrieved_ids = chroma_results['ids'][0]
                context = "\n\n".join(chroma_results['documents'][0])
            except Exception as db_error:
                # Fallback to file-based retrieval
                st.warning(f"ChromaDB query failed, using file backup: {str(db_error)}")
                result['metrics']['used_backup'] = True
            
                ids, embeddings = self._load_vectors_from_file("naive_rag")
                if embeddings is None:
                    raise Exception("Both ChromaDB and file backup failed")
                
                # Calculate similarities manually
                similarities = cosine_similarity([question_embedding], embeddings)[0]
                top_k_indices = np.argsort(similarities)[::-1][:k]
                retrieved_ids = ids[top_k_indices]
            
                # Get documents from ChromaDB by ID (if available) or reconstruct
                try:
                    chroma_results = self.chroma_collection.get(ids=retrieved_ids.tolist())
                    context = "\n\n".join(chroma_results['documents'])
                except:
                    # If we can't get from ChromaDB, use what we have from files
                    context = "Context available but not displayed in backup mode"
        
            retrieval_end_mem = psutil.virtual_memory().used / (1024 * 1024)
            retrieval_mem = retrieval_end_mem - retrieval_start_mem
            retrieval_time = time.time() - retrieval_start_time
            result['context'] = context
        
            # Prepare for precision@k and recall@k
            relevant_chunks = self._get_relevant_chunks(question, ground_truth, retrieved_ids)
        
            # Measure generation
            gen_start_time = time.time()
            gen_start_mem = psutil.virtual_memory().used / (1024 * 1024)
        
            prompt = f"""Use the following context to answer the question:
        
            Context:
            {context}
        
            Question: {question}
        
            Answer:"""
        
            response = self.gemini.generate_content(prompt)
            answer = response.text
            result['answer'] = answer
        
            gen_end_mem = psutil.virtual_memory().used / (1024 * 1024)
            gen_mem = gen_end_mem - gen_start_mem
            gen_time = time.time() - gen_start_time
        
            # Calculate metrics
            precision_at_k, recall_at_k = self._calculate_precision_recall_at_k(
                retrieved_ids, relevant_chunks, k)
        
            bleu_score = self._calculate_bleu(answer, ground_truth)
            f1 = self._calculate_f1(answer, ground_truth)
            hallucination_rate = self._detect_hallucinations(answer, context)
        
            # Update results
            result['metrics'].update({
                "precision@k": precision_at_k,
                "recall@k": recall_at_k,
                "f1_score": f1,
                "bleu_score": bleu_score,
                "retrieval_latency_ms": retrieval_time * 1000,
                "generation_latency_ms": gen_time * 1000,
                "end_to_end_latency_ms": (retrieval_time + gen_time) * 1000,
                "retrieval_memory_mb": retrieval_mem,
                "generation_memory_mb": gen_mem,
                "hallucination_rate": hallucination_rate
            })
        
        except Exception as e:
            st.error(f"Naive RAG evaluation failed: {str(e)}")
            result['answer'] = f"Error: {str(e)}"
            # If we have partial results (like retrieval succeeded but generation failed)
            if 'retrieval_time' in locals():
                result['metrics'].update({
                    "retrieval_latency_ms": retrieval_time * 1000,
                    "retrieval_memory_mb": retrieval_mem
                })
        
        return result
    
    def _evaluate_graph_rag(self, question, ground_truth, k):
        """Evaluate Graph RAG with detailed metrics"""
        result = {
            "answer": "",
            "context": "",
            "metrics": {}
        }
        
        if not self.neo4j_driver:
            result['answer'] = "Graph RAG not available (Neo4j not configured)"
            return result
            
        try:
            # Measure retrieval
            start_time = time.time()
            question_embedding = self.embedding_model.encode(question).tolist()
            
            # Memory measurement for retrieval
            retrieval_start_mem = psutil.virtual_memory().used / (1024 * 1024)
            
            with self.neo4j_driver.session() as session:
                result_neo4j = session.run("""
                MATCH (c:Chunk)
                RETURN c.id AS id, c.text AS text, c.embedding AS embedding
                """)
                
                chunks = []
                for record in result_neo4j:
                    if record["embedding"]:
                        chunks.append({
                            "id": record["id"],
                            "text": record["text"],
                            "embedding": record["embedding"]
                        })
            
            # Compute similarity in Python
            embeddings_matrix = np.array([chunk["embedding"] for chunk in chunks])
            similarities = cosine_similarity([question_embedding], embeddings_matrix)[0]
            
            top_k_indices = np.argsort(similarities)[::-1][:k]
            top_chunks = [chunks[i] for i in top_k_indices]
            context = "\n\n".join([chunk["text"] for chunk in top_chunks])
            result['context'] = context
            
            retrieval_end_mem = psutil.virtual_memory().used / (1024 * 1024)
            retrieval_mem = retrieval_end_mem - retrieval_start_mem
            retrieval_time = time.time() - start_time
            
            # Prepare for precision@k and recall@k
            relevant_chunks = self._get_relevant_chunks(question, ground_truth, [chunk["id"] for chunk in top_chunks])
            
            # Measure generation
            gen_start_time = time.time()
            gen_start_mem = psutil.virtual_memory().used / (1024 * 1024)
            
            prompt = f"""Use the following context to answer the question:
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
            
            response = self.gemini.generate_content(prompt)
            answer = response.text
            result['answer'] = answer
            
            gen_end_mem = psutil.virtual_memory().used / (1024 * 1024)
            gen_mem = gen_end_mem - gen_start_mem
            gen_time = time.time() - gen_start_time
            
            # Calculate metrics
            precision_at_k, recall_at_k = self._calculate_precision_recall_at_k(
                [chunk["id"] for chunk in top_chunks], relevant_chunks, k)
            
            bleu_score = self._calculate_bleu(answer, ground_truth)
            f1 = self._calculate_f1(answer, ground_truth)
            hallucination_rate = self._detect_hallucinations(answer, context)
            
            result['metrics'] = {
                "precision@k": precision_at_k,
                "recall@k": recall_at_k,
                "f1_score": f1,
                "bleu_score": bleu_score,
                "retrieval_latency_ms": retrieval_time * 1000,
                "generation_latency_ms": gen_time * 1000,
                "end_to_end_latency_ms": (retrieval_time + gen_time) * 1000,
                "retrieval_memory_mb": retrieval_mem,
                "generation_memory_mb": gen_mem,
                "hallucination_rate": hallucination_rate
            }
            
        except Exception as e:
            st.error(f"Graph RAG evaluation failed: {str(e)}")
            result['answer'] = f"Error: {str(e)}"
            
        return result
    
    def _evaluate_hybrid_rag(self, question, ground_truth, k=3):
        """Evaluate Hybrid RAG with detailed metrics"""
        result = {
            "answer": "",
            "context": "",
            "metrics": {}
        }
    
        try:
            # Measure end-to-end performance
            start_time = time.time()
            start_mem = psutil.virtual_memory().used / (1024 * 1024)
        
            # Query the Hybrid RAG
            answer, context = self.query_hybrid_rag(question, k)
            result['answer'] = answer
            result['context'] = context
        
            end_mem = psutil.virtual_memory().used / (1024 * 1024)
            total_mem = end_mem - start_mem
            total_time = time.time() - start_time
        
            # Get retrieved IDs from ChromaDB portion
            chroma_ids = self.chroma_collection.query(
                query_embeddings=[self.embedding_model.encode(question).tolist()],
                n_results=k
            )['ids'][0]
        
            # Get relevant chunks considering both sources
            relevant_chunks = self._get_relevant_chunks(question, ground_truth, chroma_ids)
        
            # Calculate metrics
            precision_at_k, recall_at_k = self._calculate_precision_recall_at_k(
                chroma_ids, relevant_chunks, k)
        
            bleu_score = self._calculate_bleu(answer, ground_truth)
            f1 = self._calculate_f1(answer, ground_truth)
            hallucination_rate = self._detect_hallucinations(answer, context)
        
            result['metrics'] = {
                "precision@k": precision_at_k,
                "recall@k": recall_at_k,
                "f1_score": f1,
                "bleu_score": bleu_score,
                "retrieval_latency_ms": total_time * 1000,
                "generation_latency_ms": 0,  # Combined measurement
                "end_to_end_latency_ms": total_time * 1000,
                "retrieval_memory_mb": total_mem,
                "generation_memory_mb": 0,    # Combined measurement
                "hallucination_rate": hallucination_rate
            }
        
        except Exception as e:
            st.error(f"Hybrid RAG evaluation failed: {str(e)}")
            result['answer'] = f"Error: {str(e)}"
        
        return result
    
    def _calculate_precision_recall_at_k(self, retrieved_ids, relevant_ids, k):
        """Calculate precision@k and recall@k for retrieval"""
        if not relevant_ids:
            return 0.0, 0.0
            
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        true_positives = len(retrieved_at_k & relevant_set)
        
        precision = true_positives / k if k > 0 else 0.0
        recall = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0.0
        
        return precision, recall
    
    def _get_relevant_chunks(self, question, ground_truth, all_chunk_ids):
        """Simulate relevant chunks based on ground truth (in a real system, this would be labeled data)"""
        # This is a simplified approach - in reality you'd need labeled data
        # Here we assume chunks containing ground truth words are relevant
        if not ground_truth.strip():
            return []
            
        ground_truth_words = set(word_tokenize(ground_truth.lower()))
        relevant_chunks = []
        
        for chunk_id in all_chunk_ids:
            chunk_text = self._get_chunk_text(chunk_id)
            if not chunk_text:
                continue
                
            chunk_words = set(word_tokenize(chunk_text.lower()))
            if ground_truth_words & chunk_words:  # intersection
                relevant_chunks.append(chunk_id)
                
        return relevant_chunks
    
    def _get_chunk_text(self, chunk_id):
        """Get chunk text by ID from ChromaDB"""
        try:
            result = self.chroma_collection.get(ids=[chunk_id])
            return result['documents'][0] if result['documents'] else ""
        except:
            return ""
    
    def _calculate_bleu(self, candidate, reference):
        """Calculate BLEU score between candidate and reference"""
        if not reference.strip() or not candidate.strip():
            return 0.0
            
        smoothie = SmoothingFunction().method4
        candidate_tokens = word_tokenize(candidate.lower())
        reference_tokens = word_tokenize(reference.lower())
        
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    
    def _calculate_f1(self, candidate, reference):
        """Calculate F1 score between candidate and reference"""
        if not reference.strip() or not candidate.strip():
            return 0.0
            
        candidate_tokens = set(word_tokenize(candidate.lower()))
        reference_tokens = set(word_tokenize(reference.lower()))
        
        if not reference_tokens:
            return 0.0
            
        common_tokens = candidate_tokens & reference_tokens
        precision = len(common_tokens) / len(candidate_tokens) if candidate_tokens else 0.0
        recall = len(common_tokens) / len(reference_tokens)
        
        if (precision + recall) == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
    
    def _detect_hallucinations(self, answer, context):
        """Simple hallucination detection (checks if answer contains facts not in context)"""
        if not answer.strip() or not context.strip():
            return 0.0
            
        answer_sents = nltk.sent_tokenize(answer)
        context_sents = nltk.sent_tokenize(context)
        
        if not answer_sents:
            return 0.0
            
        # Simple approach: count sentences not similar to any context sentence
        hallucination_count = 0
        for a_sent in answer_sents:
            is_hallucination = True
            a_tokens = set(word_tokenize(a_sent.lower()))
            
            for c_sent in context_sents:
                c_tokens = set(word_tokenize(c_sent.lower()))
                if a_tokens & c_tokens:  # if any overlap
                    is_hallucination = False
                    break
                    
            if is_hallucination:
                hallucination_count += 1
                
        return hallucination_count / len(answer_sents)
    
    def _answers_match(self, answer1, answer2):
        """Check if two answers are similar enough"""
        return self._calculate_bleu(answer1, answer2) > 0.4 or self._calculate_f1(answer1, answer2) > 0.4

def main():
    st.set_page_config(page_title="RAG Evaluation", layout="wide")
    st.title("RAG Performance Evaluation")
    st.write("Compare Naive RAG, Graph RAG, and Hybrid RAG approaches")
    
    # Initialize RAG systems
    if 'rag' not in st.session_state:
        try:
            st.session_state.rag = RAGSystems()
        except Exception as e:
            st.error(f"Failed to initialize RAG systems: {str(e)}")
            st.stop()
    
    # Document upload section
    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or text files (up to 3)", 
        type=['pdf', 'txt'], 
        accept_multiple_files=True,
        key="file_uploader")
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing documents..."):
            st.session_state.rag.process_documents(uploaded_files, clear_existing=True)
    
    # Query section
    st.subheader("2. Ask a Question")
    question = st.text_input("Enter your question about the documents:", key="question")
    ground_truth = st.text_area("Enter the ground truth answer (for evaluation):", key="ground_truth")
    top_k = st.slider("Select top k chunks for retrieval:", 1, 10, 3)
    st.subheader("🔍 Vector Inspection Tools")

    if st.checkbox("Show vector statistics"):
        st.session_state.rag.display_vector_stats()

    if st.checkbox("Inspect specific vector"):
        chunk_id = st.text_input("Enter chunk ID (e.g., 'document1_0'):")
        if chunk_id:
            st.session_state.rag.display_vector_by_id(chunk_id)

    if st.checkbox("Show random sample vectors"):
        num_samples = st.slider("Number of samples", 1, 5, 2)
        st.session_state.rag.display_random_vectors(num_samples)
    if st.button("Evaluate RAG Systems") and question:
        if not ground_truth:
            st.warning("Evaluation will be less accurate without ground truth")
        
        with st.spinner("Evaluating RAG systems..."):
            results = st.session_state.rag.evaluate_performance(question, ground_truth, top_k)
            
            # Display answers
            st.subheader("3. Results from Different RAG Systems")
            cols = st.columns(3)
            
            for idx, (system, data) in enumerate(results.items()):
                with cols[idx]:
                    st.markdown(f"### {system}")
                    st.markdown(f"**Answer:** {data['answer']}")
                    
                    # Display metrics in an expandable section
                    with st.expander("View Metrics"):
                        st.markdown("**Performance Metrics:**")
                        st.write(f"Precision@{top_k}: {data['metrics']['precision@k']:.2f}")
                        st.write(f"Recall@{top_k}: {data['metrics']['recall@k']:.2f}")
                        st.write(f"F1 Score: {data['metrics']['f1_score']:.2f}")
                        st.write(f"BLEU Score: {data['metrics']['bleu_score']:.2f}")
                        st.write(f"Hallucination Rate: {data['metrics']['hallucination_rate']:.2f}")
                        
                        st.markdown("**Latency Metrics:**")
                        st.write(f"Retrieval: {data['metrics']['retrieval_latency_ms']:.2f} ms")
                        st.write(f"Generation: {data['metrics']['generation_latency_ms']:.2f} ms")
                        st.write(f"End-to-End: {data['metrics']['end_to_end_latency_ms']:.2f} ms")
                        
                        st.markdown("**Memory Usage:**")
                        st.write(f"Retrieval: {data['metrics']['retrieval_memory_mb']:.2f} MB")
                        st.write(f"Generation: {data['metrics']['generation_memory_mb']:.2f} MB")
                    
                    # Context viewer
                    if st.button(f"View Context for {system}"):
                        st.text_area("Context", data['context'], height=200, key=f"context_{system}")
            
            # Display performance metrics comparison
            st.subheader("4. Performance Metrics Comparison")
            
            # Create metrics tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Retrieval Metrics**")
                retrieval_data = {
                    "System": list(results.keys()),
                    f"Precision@{top_k}": [results[sys]["metrics"]["precision@k"] for sys in results],
                    f"Recall@{top_k}": [results[sys]["metrics"]["recall@k"] for sys in results],
                    "Hallucination Rate": [results[sys]["metrics"]["hallucination_rate"] for sys in results]
                }
                st.dataframe(retrieval_data)
                
            with col2:
                st.markdown("**Quality Metrics**")
                quality_data = {
                    "System": list(results.keys()),
                    "F1 Score": [results[sys]["metrics"]["f1_score"] for sys in results],
                    "BLEU Score": [results[sys]["metrics"]["bleu_score"] for sys in results]
                }
                st.dataframe(quality_data)
            
            # Latency and memory comparison
            st.markdown("**Latency and Memory (Lower is Better)**")
            perf_data = {
                "System": list(results.keys()),
                "End-to-End Latency (ms)": [results[sys]["metrics"]["end_to_end_latency_ms"] for sys in results],
                "Retrieval Memory (MB)": [results[sys]["metrics"]["retrieval_memory_mb"] for sys in results],
                "Generation Memory (MB)": [results[sys]["metrics"]["generation_memory_mb"] for sys in results]
            }
            st.dataframe(perf_data)
            
            # Visualization
            st.subheader("5. Metrics Visualization")
            
            # Select which metrics to visualize
            metrics_options = [
                f"Precision@{top_k}", f"Recall@{top_k}", "F1 Score", 
                "BLEU Score", "Hallucination Rate",
                "End-to-End Latency (ms)"
            ]
            selected_metrics = st.multiselect(
                "Select metrics to visualize", 
                metrics_options,
                default=[f"Precision@{top_k}", f"Recall@{top_k}", "F1 Score"]
            )
            
            if selected_metrics:
                chart_data = {"System": list(results.keys())}
                for metric in selected_metrics:
                    if metric in [f"Precision@{top_k}", f"Recall@{top_k}", "F1 Score", "BLEU Score", "Hallucination Rate"]:
                        chart_data[metric] = [results[sys]["metrics"][metric.lower().replace(f"@{top_k}", "@k").replace(" (ms)", "").replace(" ", "_")] for sys in results]
                    elif metric == "End-to-End Latency (ms)":
                        chart_data[metric] = [results[sys]["metrics"]["end_to_end_latency_ms"] for sys in results]
                
                st.bar_chart(chart_data, x="System")


if __name__ == "__main__":
    main()