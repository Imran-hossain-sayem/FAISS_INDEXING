import streamlit as st
import pandas as pd
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Book Search with FAISS Indexing",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 FAISS Indexing Dashboard")
st.markdown("""
This dashboard demonstrates different FAISS indexing methods for semantic book search using sentence embeddings.
""")

# Initialize data
@st.cache_resource
def load_data():
    books = [
        {"book_name": "The Double Helix", "summary": "tells the story of the discovery of DNA, which is one of the most significant scientific findings in all"},
        {"book_name": "iWoz", "summary": "is Steve Wozniak's autobiography, detailing his story in his own words, from early tinkering with electro"},
        {"book_name": "The Truths We Hold", "summary": "is the autobiography of civil rights activist, Californian Senator, and Vice President Kamala Harris, whi"},
        {"book_name": "Farmageddon", "summary": "is a shocking compendium of the facts and figures about how the mass production of cheap meat influences"},
        {"book_name": "Excellent Sheep", "summary": "describes how fundamentally broken elite education is, why it makes students feel depressed and lost, how"},
        {"book_name": "The Omnivore's Dilemma", "summary": "explains the range of food choices we face today using four meals on a spectrum from highly processed to"},
        {"book_name": "Surrounded by Idiots", "summary": "offers great advice on how to get your point across more effectively, communicate better, and work your"},
        {"book_name": "The Alchemist", "summary": "is a classic novel in which a boy named Santiago embarks on a journey seeking treasure in the Egyptian py"},
        {"book_name": "Words That Work", "summary": "outlines the importance of using the right words and the appropriate body language in a given situation t"},
        {"book_name": "Think Again", "summary": "will make you more intelligent, persuasive, and self-aware by identifying the power of being humble about"},
        {"book_name": "No Logo", "summary": "uses four parts, including 'No Space,' 'No Choice,' 'No Jobs,' and 'No Logo,' to explain the growth of br"},
        {"book_name": "Mind Hacking", "summary": "is a hands-on guide on how to transform your mind in just 21 days, which is the time required for your br"},
        {"book_name": "Super Human", "summary": "presents the groundbreaking discoveries of Dave Asprey (the CEO of Bulletproof) in the field of diet & nu"},
        {"book_name": "Don Quixote", "summary": "is a classic novel from 1605 which portraits the life and insightful journey of Don Quixote de la Mancha"},
        {"book_name": "Inspired", "summary": "taps into a popular subject, which is how to build successful products that sell, run a thriving business"},
        {"book_name": "Ego Is The Enemy", "summary": "reveals why a tendency that's hardwired into our brains — the belief that the world revolves around us an"},
        {"book_name": "Siddhartha", "summary": "presents the self-discovery expedition of a man during the time of the Buddha who, unsure of what life re"},
        {"book_name": "The Everything Store", "summary": "is the closest biographical documentation of the unprecedented rise of Amazon as an online retail store w"},
        {"book_name": "Bold", "summary": "shows you that exponential technology has democratized the power to change the world and build wealth, by"},
        {"book_name": "The Shallows", "summary": "explores the effects of the Internet on the human brain, which aren't entirely positive, as our constant"}
    ]
    
    # Category mapping
    categories = {
        "The Double Helix": "science",
        "iWoz": "biography",
        "The Truths We Hold": "politics",
        "Farmageddon": "environment",
        "Excellent Sheep": "psychology",
        "The Omnivore's Dilemma": "economics",
        "Surrounded by Idiots": "psychology",
        "The Alchemist": "fiction",
        "Words That Work": "marketing",
        "Think Again": "psychology",
        "No Logo": "business",
        "Mind Hacking": "motivation",
        "Super Human": "health",
        "Don Quixote": "fiction",
        "Inspired": "management",
        "Ego Is The Enemy": "creativity",
        "Siddhartha": "fiction",
        "The Everything Store": "business",
        "Bold": "motivation",
        "The Shallows": "psychology"
    }
    
    for book in books:
        book['category'] = categories.get(book['book_name'], 'other')
    
    return books

books = load_data()

# Create texts and embeddings
@st.cache_resource
def create_embeddings(books):
    texts = [f"{book['book_name']}. {book['summary']}" for book in books]
    book_names = [book['book_name'] for book in books]
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    return texts, book_names, embeddings, model

texts, book_names, embeddings, model = create_embeddings(books)

N, D = embeddings.shape

# Create FAISS indices
@st.cache_resource
def create_indices(embeddings, D):
    indices = {}
    times = {}
    
    # KNN (Ground Truth)
    start = time.perf_counter()
    knn_index = faiss.IndexFlatL2(D)
    knn_index.add(embeddings)
    times['knn'] = (time.perf_counter() - start) * 1000
    indices['knn'] = knn_index
    
    # IVF
    nlist = 4
    quantizer = faiss.IndexFlatL2(D)
    ivf_index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)
    start = time.perf_counter()
    ivf_index.train(embeddings)
    ivf_index.add(embeddings)
    times['ivf_training'] = (time.perf_counter() - start) * 1000
    ivf_index.nprobe = 2
    indices['ivf'] = ivf_index
    
    # PQ
    M = 8
    nbits = 4
    pq_index = faiss.IndexPQ(D, M, nbits)
    start = time.perf_counter()
    pq_index.train(embeddings)
    pq_index.add(embeddings)
    times['pq_training'] = (time.perf_counter() - start) * 1000
    indices['pq'] = pq_index
    
    # IVF-PQ
    ivfpq_index = faiss.IndexIVFPQ(quantizer, D, nlist, M, nbits)
    start = time.perf_counter()
    ivfpq_index.train(embeddings)
    ivfpq_index.add(embeddings)
    times['ivfpq_training'] = (time.perf_counter() - start) * 1000
    ivfpq_index.nprobe = 2
    indices['ivfpq'] = ivfpq_index
    
    # HNSW
    hnsw_index = faiss.IndexHNSWFlat(D, 32)
    start = time.perf_counter()
    hnsw_index.add(embeddings)
    times['hnsw'] = (time.perf_counter() - start) * 1000
    indices['hnsw'] = hnsw_index
    
    return indices, times

indices, index_times = create_indices(embeddings, D)

# Sidebar
st.sidebar.header("🔍 Search Settings")
user_query = st.sidebar.text_input("Enter your book query:", value="A book about science")
K = st.sidebar.slider("Number of books to retrieve (K):", min_value=1, max_value=10, value=5)
search_button = st.sidebar.button("🔍 Search", type="primary")

st.sidebar.markdown("---")
st.sidebar.header("📊 Index Info")
st.sidebar.write(f"Number of books: {N}")
st.sidebar.write(f"Embedding dimension: {D}")

# Main content
if search_button or user_query:
    # Encode query
    query_embedding = model.encode([user_query], convert_to_numpy=True).astype("float32")
    
    # Search with all indices
    results = {}
    query_times = {}
    
    for name, index in indices.items():
        if name in ['knn', 'hnsw']:
            start = time.perf_counter()
            distances, indices_result = index.search(query_embedding, K)
            query_times[name] = (time.perf_counter() - start) * 1000
            results[name] = {
                'indices': indices_result[0],
                'distances': distances[0],
                'books': [book_names[i] for i in indices_result[0]]
            }
        else:  # IVF, PQ, IVFPQ
            start = time.perf_counter()
            distances, indices_result = index.search(query_embedding, K)
            query_times[name] = (time.perf_counter() - start) * 1000
            results[name] = {
                'indices': indices_result[0],
                'distances': distances[0],
                'books': [book_names[i] for i in indices_result[0]]
            }
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📖 Search Results")
        
        # Create tabs for different indices
        tabs = st.tabs(["KNN (Ground Truth)", "IVF", "PQ", "IVF-PQ", "HNSW"])
        
        for tab, (name, result) in zip(tabs, results.items()):
            with tab:
                df = pd.DataFrame({
                    'Rank': range(1, K+1),
                    'Book Name': result['books'],
                    'Distance': [f"{d:.4f}" for d in result['distances']]
                })
                st.dataframe(df, use_container_width=True)
                
                # Show categories
                categories_found = [books[book_names.index(b)]['category'] for b in result['books']]
                st.caption(f"Categories: {', '.join(set(categories_found))}")
    
    with col2:
        st.subheader("⚡ Performance Metrics")
        
        # Query time comparison
        fig = go.Figure(data=[
            go.Bar(
                x=list(query_times.keys()),
                y=list(query_times.values()),
                text=[f"{t:.2f}ms" for t in query_times.values()],
                textposition='auto',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            )
        ])
        fig.update_layout(
            title="Query Time Comparison",
            xaxis_title="Index Type",
            yaxis_title="Time (ms)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Precision, Recall, and Accuracy metrics
        knn_set = set(results['knn']['indices'])
        
        metrics = {}
        for name, result in results.items():
            if name != 'knn':
                pred_set = set(result['indices'])
                
                # Calculate Precision@K
                relevant_retrieved = len(knn_set & pred_set)
                precision = relevant_retrieved / len(pred_set) if len(pred_set) > 0 else 0
                
                # Calculate Recall@K
                recall = relevant_retrieved / len(knn_set) if len(knn_set) > 0 else 0
                
                # Calculate Accuracy@K (whether the set of retrieved items exactly matches ground truth)
                accuracy = 1 if knn_set == pred_set else 0
                
                metrics[name] = {
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy
                }
        
        if metrics:
            df_metrics = pd.DataFrame(metrics).T
            df_metrics.columns = ['Precision@K', 'Recall@K', 'Accuracy@K']
            df_metrics = df_metrics.round(3)
            
            # Highlight the metrics
            st.dataframe(
                df_metrics.style.background_gradient(cmap='Blues', subset=['Precision@K', 'Recall@K']),
                use_container_width=True
            )
            
            # Add a note about the metrics
            st.caption("""
            **Metrics Explanation:**
            - **Precision@K**: Fraction of retrieved items that are relevant (in ground truth)
            - **Recall@K**: Fraction of relevant items that were retrieved
            - **Accuracy@K**: Whether the retrieved set exactly matches the ground truth
            """)
    
    # Category Distribution
    st.subheader("📊 Category Distribution of Results")
    
    # Get all categories from top results
    all_categories = []
    for name, result in results.items():
        categories_found = [books[book_names.index(b)]['category'] for b in result['books']]
        all_categories.extend(categories_found)
    
    if all_categories:
        cat_counts = pd.Series(all_categories).value_counts()
        fig = go.Figure(data=[
            go.Pie(labels=cat_counts.index, values=cat_counts.values, hole=0.3)
        ])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Add Recall@K visualization
    st.subheader("📈 Recall@K Comparison")
    
    # Create a bar chart for Recall@K
    recall_data = {}
    for name, metric in metrics.items():
        recall_data[name] = metric['recall']
    
    if recall_data:
        fig_recall = go.Figure(data=[
            go.Bar(
                x=list(recall_data.keys()),
                y=list(recall_data.values()),
                text=[f"{r:.3f}" for r in recall_data.values()],
                textposition='auto',
                marker_color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            )
        ])
        fig_recall.update_layout(
            title="Recall@K by Index Type",
            xaxis_title="Index Type",
            yaxis_title="Recall@K",
            yaxis_range=[0, 1],
            height=300
        )
        st.plotly_chart(fig_recall, use_container_width=True)

# Add information at bottom
st.markdown("---")
st.markdown("""
**About this dashboard:**
- Uses FAISS for efficient similarity search on book embeddings
- Compares different indexing methods: KNN (ground truth), IVF, PQ, IVF-PQ, and HNSW
- All embeddings are generated using the `all-MiniLM-L6-v2` model from Sentence Transformers
- **Precision@K**: Measures accuracy of retrieved results
- **Recall@K**: Measures completeness of retrieved results
""")

# Display all books in expander
with st.expander("📚 View All Books"):
    df_books = pd.DataFrame(books)
    st.dataframe(df_books, use_container_width=True)
