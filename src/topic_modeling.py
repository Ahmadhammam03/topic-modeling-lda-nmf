"""
Topic Modeling Module - FIXED VERSION

This module provides implementations for both Latent Dirichlet Allocation (LDA)
and Non-Negative Matrix Factorization (NMF) topic modeling algorithms.

Author: Ahmad Hammam
GitHub: @Ahmadhammam03
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TopicModelingPipeline:
    """
    A comprehensive topic modeling pipeline supporting both LDA and NMF algorithms.
    """
    
    def __init__(self, algorithm: str = 'lda', n_topics: int = 7, random_state: int = 42):
        """
        Initialize the topic modeling pipeline.
        
        Args:
            algorithm (str): Algorithm to use ('lda' or 'nmf')
            n_topics (int): Number of topics to discover
            random_state (int): Random state for reproducibility
        """
        self.algorithm = algorithm.lower()
        self.n_topics = n_topics
        self.random_state = random_state
        
        # Initialize components
        self.vectorizer = None
        self.model = None
        self.document_term_matrix = None
        self.feature_names = None
        self.topics = {}
        
        # Validate algorithm choice
        if self.algorithm not in ['lda', 'nmf']:
            raise ValueError("Algorithm must be 'lda' or 'nmf'")
    
    def _setup_vectorizer(self, vectorizer_type: str = 'auto', **kwargs) -> None:
        """Setup the text vectorizer based on the algorithm."""
        # Default parameters
        default_params = {
            'max_df': 0.95,
            'min_df': 2,
            'stop_words': 'english'
        }
        default_params.update(kwargs)
        
        # Choose vectorizer based on algorithm or user preference
        if vectorizer_type == 'auto':
            if self.algorithm == 'lda':
                self.vectorizer = CountVectorizer(**default_params)
            else:  # nmf
                self.vectorizer = TfidfVectorizer(**default_params)
        elif vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(**default_params)
        elif vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(**default_params)
        else:
            raise ValueError("vectorizer_type must be 'auto', 'count', or 'tfidf'")
    
    def _setup_model(self) -> None:
        """Setup the topic modeling algorithm."""
        if self.algorithm == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=10
            )
        else:  # nmf
            self.model = NMF(
                n_components=self.n_topics,
                random_state=self.random_state
            )
    
    def fit(self, documents: List[str], vectorizer_type: str = 'auto', **vectorizer_kwargs) -> 'TopicModelingPipeline':
        """
        Fit the topic modeling pipeline to documents.
        
        Args:
            documents (List[str]): List of text documents
            vectorizer_type (str): Type of vectorizer to use
            **vectorizer_kwargs: Additional parameters for vectorizer
            
        Returns:
            TopicModelingPipeline: Self for method chaining
        """
        print(f"🚀 Starting {self.algorithm.upper()} topic modeling with {self.n_topics} topics...")
        
        # Setup vectorizer and model
        self._setup_vectorizer(vectorizer_type, **vectorizer_kwargs)
        self._setup_model()
        
        # Vectorize documents
        print("📝 Vectorizing documents...")
        self.document_term_matrix = self.vectorizer.fit_transform(documents)
        
        # Get feature names (try both old and new API)
        try:
            self.feature_names = self.vectorizer.get_feature_names_out()
        except AttributeError:
            try:
                self.feature_names = self.vectorizer.get_feature_names()
            except AttributeError:
                # Fallback for very old sklearn versions
                self.feature_names = self.vectorizer.vocabulary_.keys()
        
        # Fit the topic model
        print("🔍 Discovering topics...")
        self.model.fit(self.document_term_matrix)
        
        # Extract topics
        self._extract_topics()
        
        print(f"✅ Topic modeling completed! Discovered {self.n_topics} topics.")
        return self
    
    def _extract_topics(self, top_words: int = 15) -> None:
        """
        Extract and interpret topics from the trained model.
        
        Args:
            top_words (int): Number of top words to extract per topic
        """
        for topic_idx, topic in enumerate(self.model.components_):
            # Get top word indices
            top_word_indices = topic.argsort()[-top_words:][::-1]
            top_words_list = [self.feature_names[i] for i in top_word_indices]
            
            # Store topic information
            self.topics[topic_idx] = {
                'words': top_words_list,
                'weights': [topic[i] for i in top_word_indices],
                'name': self._generate_topic_name(top_words_list[:5])
            }
    
    def _generate_topic_name(self, top_words: List[str]) -> str:
        """
        Generate a human-readable name for a topic based on top words.
        
        Args:
            top_words (List[str]): Top words for the topic
            
        Returns:
            str: Generated topic name
        """
        # Safety check for empty words list
        if not top_words:
            return "Unknown Topic"
        
        # Topic name mapping based on common word patterns
        topic_patterns = {
            ('trump', 'president', 'election', 'political', 'campaign'): 'Politics & Elections',
            ('health', 'medical', 'disease', 'patients', 'study'): 'Healthcare & Medicine',
            ('school', 'education', 'students', 'learning', 'university'): 'Education & Learning',
            ('business', 'company', 'money', 'market', 'financial'): 'Business & Finance',
            ('technology', 'computer', 'internet', 'digital', 'software'): 'Technology',
            ('love', 'relationship', 'family', 'personal', 'life'): 'Relationships & Life',
            ('music', 'art', 'cultural', 'entertainment', 'creative'): 'Arts & Culture',
            ('food', 'cooking', 'recipe', 'kitchen', 'meal'): 'Food & Cooking',
            ('travel', 'world', 'country', 'place', 'visit'): 'Travel & Places',
            ('science', 'research', 'study', 'scientific', 'data'): 'Science & Research'
        }
        
        # Check for pattern matches
        top_words_lower = [word.lower() for word in top_words]
        for pattern, name in topic_patterns.items():
            if any(word in top_words_lower for word in pattern):
                return name
        
        # Default naming based on top words (with safety checks)
        if len(top_words) >= 2:
            return f"{top_words[0].title()} & {top_words[1].title()}"
        elif len(top_words) == 1:
            return f"{top_words[0].title()} Topic"
        else:
            return "General Topic"
    
    def transform(self, documents: Optional[List[str]] = None) -> np.ndarray:
        """
        Transform documents into topic space.
        
        Args:
            documents (List[str], optional): Documents to transform. If None, uses training documents.
            
        Returns:
            np.ndarray: Document-topic matrix
        """
        if self.model is None:
            raise ValueError("Model must be fitted before transformation")
        
        if documents is not None:
            dtm = self.vectorizer.transform(documents)
        else:
            dtm = self.document_term_matrix
        
        return self.model.transform(dtm)
    
    def predict_topics(self, documents: Optional[List[str]] = None) -> List[int]:
        """
        Predict the primary topic for each document.
        
        Args:
            documents (List[str], optional): Documents to predict. If None, uses training documents.
            
        Returns:
            List[int]: Predicted topic indices
        """
        topic_probs = self.transform(documents)
        return topic_probs.argmax(axis=1)
    
    def get_topic_distribution(self, document: str) -> Dict[str, float]:
        """
        Get topic distribution for a single document.
        
        Args:
            document (str): Input document
            
        Returns:
            Dict[str, float]: Topic names and their probabilities
        """
        doc_topics = self.transform([document])[0]
        
        distribution = {}
        for topic_idx, prob in enumerate(doc_topics):
            topic_name = self.topics[topic_idx]['name']
            distribution[topic_name] = prob
        
        return distribution
    
    def print_topics(self, num_words: int = 10) -> None:
        """
        Print discovered topics in a formatted way.
        
        Args:
            num_words (int): Number of words to display per topic
        """
        print(f"\n🎯 Discovered Topics ({self.algorithm.upper()}):")
        print("=" * 60)
        
        for topic_idx, topic_info in self.topics.items():
            words = topic_info['words'][:num_words]
            name = topic_info['name']
            
            print(f"\n📌 Topic {topic_idx}: {name}")
            print(f"   Top words: {', '.join(words)}")
    
    def save_results(self, filepath: str) -> None:
        """
        Save topic modeling results to JSON file.
        
        Args:
            filepath (str): Path to save the results
        """
        import json
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        results = {
            'algorithm': self.algorithm,
            'n_topics': self.n_topics,
            'topics': self.topics,
            'model_params': {
                'random_state': self.random_state
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to {filepath}")


class TopicModelComparison:
    """
    Compare LDA and NMF topic modeling results.
    """
    
    def __init__(self, documents: List[str], n_topics: int = 7, random_state: int = 42):
        """
        Initialize comparison with documents.
        
        Args:
            documents (List[str]): Documents to analyze
            n_topics (int): Number of topics for both algorithms
            random_state (int): Random state for reproducibility
        """
        self.documents = documents
        self.n_topics = n_topics
        self.random_state = random_state
        
        # Initialize models
        self.lda_model = TopicModelingPipeline('lda', n_topics, random_state)
        self.nmf_model = TopicModelingPipeline('nmf', n_topics, random_state)
    
    def fit_both(self) -> 'TopicModelComparison':
        """
        Fit both LDA and NMF models.
        
        Returns:
            TopicModelComparison: Self for method chaining
        """
        print("🔬 Comparing LDA vs NMF Topic Modeling")
        print("=" * 50)
        
        # Fit LDA
        print("\n1️⃣ Training LDA Model:")
        self.lda_model.fit(self.documents)
        
        # Fit NMF
        print("\n2️⃣ Training NMF Model:")
        self.nmf_model.fit(self.documents)
        
        return self
    
    def compare_topics(self) -> None:
        """Print side-by-side comparison of topics."""
        print("\n📊 Topic Comparison:")
        print("=" * 80)
        
        for i in range(self.n_topics):
            lda_words = self.lda_model.topics[i]['words'][:8]
            nmf_words = self.nmf_model.topics[i]['words'][:8]
            lda_name = self.lda_model.topics[i]['name']
            nmf_name = self.nmf_model.topics[i]['name']
            
            print(f"\n🔍 Topic {i}:")
            print(f"  LDA ({lda_name}):")
            print(f"    {', '.join(lda_words)}")
            print(f"  NMF ({nmf_name}):")
            print(f"    {', '.join(nmf_words)}")
    
    def analyze_document(self, document: str) -> None:
        """
        Analyze a single document with both models.
        
        Args:
            document (str): Document to analyze
        """
        print(f"\n📄 Document Analysis:")
        print(f"Text preview: {document[:100]}...")
        print("\n🎯 Topic Predictions:")
        
        # LDA analysis
        lda_dist = self.lda_model.get_topic_distribution(document)
        lda_top_topic = max(lda_dist.items(), key=lambda x: x[1])
        
        # NMF analysis
        nmf_dist = self.nmf_model.get_topic_distribution(document)
        nmf_top_topic = max(nmf_dist.items(), key=lambda x: x[1])
        
        print(f"  LDA: {lda_top_topic[0]} ({lda_top_topic[1]:.3f})")
        print(f"  NMF: {nmf_top_topic[0]} ({nmf_top_topic[1]:.3f})")


def create_sample_analysis():
    """Create a sample analysis for demonstration."""
    # Sample documents for testing
    sample_docs = [
        "The president announced new healthcare policies today affecting millions of Americans.",
        "Machine learning and artificial intelligence are transforming the technology industry.",
        "Students across the country are returning to schools after summer vacation.",
        "The stock market showed significant gains following the latest economic reports.",
        "Scientists discovered a new treatment for cancer in clinical trials.",
        "The election campaign is heating up with debates scheduled for next month."
    ]
    
    print("🚀 Running Sample Topic Modeling Analysis")
    print("=" * 50)
    
    # Create and run comparison
    comparison = TopicModelComparison(sample_docs, n_topics=3, random_state=42)
    comparison.fit_both()
    comparison.compare_topics()
    
    # Analyze a sample document
    comparison.analyze_document(sample_docs[0])
    
    # Save results to results folder
    import os
    os.makedirs('results', exist_ok=True)
    
    comparison.lda_model.save_results('results/sample_lda_results.json')
    comparison.nmf_model.save_results('results/sample_nmf_results.json')
    print("\n💾 Sample results saved to results/ folder!")


if __name__ == "__main__":
    create_sample_analysis()