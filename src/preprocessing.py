"""
Text Preprocessing Module for Topic Modeling

This module provides comprehensive text preprocessing utilities optimized
for topic modeling tasks including cleaning, normalization, and feature extraction.

Author: Ahmad Hammam
GitHub: @Ahmadhammam03
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Some advanced features will be disabled.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available. Language detection will be disabled.")


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for topic modeling.
    
    This class provides a complete suite of text preprocessing tools including:
    - Text cleaning and normalization
    - Tokenization and stemming/lemmatization
    - Stopword removal with custom lists
    - Language detection and filtering
    - Statistical text analysis
    """
    
    def __init__(self, language: str = 'english', custom_stopwords: Optional[List[str]] = None):
        """
        Initialize the text preprocessor.
        
        Args:
            language (str): Language for stopwords and processing
            custom_stopwords (List[str], optional): Additional stopwords to remove
        """
        self.language = language
        self.custom_stopwords = custom_stopwords or []
        
        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            self._download_nltk_data()
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words(language))
        else:
            # Basic English stopwords if NLTK not available
            self.stop_words = set([
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above',
                'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            ])
        
        # Add custom stopwords
        self.stop_words.update(self.custom_stopwords)
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def _download_nltk_data(self) -> None:
        """Download required NLTK data."""
        try:
            required_data = ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger', 
                           'maxent_ne_chunker', 'words', 'omw-1.4']
            for item in required_data:
                try:
                    nltk.data.find(f'tokenizers/{item}')
                except LookupError:
                    try:
                        nltk.download(item, quiet=True)
                    except:
                        pass
        except Exception:
            pass
    
    def clean_text(self, text: str, 
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_mentions: bool = True,
                   remove_hashtags: bool = True,
                   remove_numbers: bool = False,
                   remove_punctuation: bool = True,
                   lowercase: bool = True,
                   remove_extra_whitespace: bool = True) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text to clean
            remove_urls (bool): Remove URLs
            remove_emails (bool): Remove email addresses
            remove_mentions (bool): Remove @mentions
            remove_hashtags (bool): Remove #hashtags
            remove_numbers (bool): Remove numbers
            remove_punctuation (bool): Remove punctuation
            lowercase (bool): Convert to lowercase
            remove_extra_whitespace (bool): Remove extra whitespace
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        if remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove emails
        if remove_emails:
            text = self.email_pattern.sub('', text)
        
        # Remove mentions
        if remove_mentions:
            text = self.mention_pattern.sub('', text)
        
        # Remove hashtags
        if remove_hashtags:
            text = self.hashtag_pattern.sub('', text)
        
        # Remove numbers
        if remove_numbers:
            text = self.number_pattern.sub('', text)
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Remove punctuation
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        if remove_extra_whitespace:
            text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def tokenize(self, text: str, method: str = 'basic') -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text
            method (str): Tokenization method ('basic', 'nltk', 'textblob')
            
        Returns:
            List[str]: List of tokens
        """
        if not text:
            return []
        
        if method == 'basic':
            return text.split()
        elif method == 'nltk' and NLTK_AVAILABLE:
            return word_tokenize(text)
        elif method == 'textblob' and TEXTBLOB_AVAILABLE:
            return TextBlob(text).words
        else:
            return text.split()
    
    def remove_stopwords(self, tokens: List[str], 
                        additional_stopwords: Optional[List[str]] = None) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens (List[str]): List of tokens
            additional_stopwords (List[str], optional): Additional stopwords to remove
            
        Returns:
            List[str]: Filtered tokens
        """
        stopwords_set = self.stop_words.copy()
        if additional_stopwords:
            stopwords_set.update(additional_stopwords)
        
        return [token for token in tokens if token.lower() not in stopwords_set]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Lemmatized tokens
        """
        if not NLTK_AVAILABLE:
            return tokens
        
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Stem tokens to their root form.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Stemmed tokens
        """
        if not NLTK_AVAILABLE:
            return tokens
        
        return [self.stemmer.stem(token) for token in tokens]
    
    def filter_tokens(self, tokens: List[str], 
                     min_length: int = 2,
                     max_length: int = 20,
                     alpha_only: bool = True) -> List[str]:
        """
        Filter tokens based on various criteria.
        
        Args:
            tokens (List[str]): List of tokens
            min_length (int): Minimum token length
            max_length (int): Maximum token length
            alpha_only (bool): Keep only alphabetic tokens
            
        Returns:
            List[str]: Filtered tokens
        """
        filtered_tokens = []
        
        for token in tokens:
            # Length filter
            if len(token) < min_length or len(token) > max_length:
                continue
            
            # Alphabetic filter
            if alpha_only and not token.isalpha():
                continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def preprocess_text(self, text: str,
                       clean_params: Optional[Dict] = None,
                       tokenize_method: str = 'basic',
                       remove_stopwords: bool = True,
                       use_lemmatization: bool = True,
                       use_stemming: bool = False,
                       filter_params: Optional[Dict] = None) -> str:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text (str): Input text
            clean_params (Dict, optional): Parameters for text cleaning
            tokenize_method (str): Tokenization method
            remove_stopwords (bool): Whether to remove stopwords
            use_lemmatization (bool): Whether to lemmatize
            use_stemming (bool): Whether to stem (applied after lemmatization)
            filter_params (Dict, optional): Parameters for token filtering
            
        Returns:
            str: Preprocessed text
        """
        # Set default parameters
        clean_params = clean_params or {}
        filter_params = filter_params or {}
        
        # Step 1: Clean text
        cleaned_text = self.clean_text(text, **clean_params)
        
        # Step 2: Tokenize
        tokens = self.tokenize(cleaned_text, tokenize_method)
        
        # Step 3: Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Step 4: Lemmatize
        if use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        
        # Step 5: Stem (if requested)
        if use_stemming:
            tokens = self.stem_tokens(tokens)
        
        # Step 6: Filter tokens
        tokens = self.filter_tokens(tokens, **filter_params)
        
        return ' '.join(tokens)
    
    def preprocess_documents(self, documents: Union[List[str], pd.Series],
                           show_progress: bool = True,
                           **preprocess_params) -> List[str]:
        """
        Preprocess a collection of documents.
        
        Args:
            documents (Union[List[str], pd.Series]): Collection of documents
            show_progress (bool): Whether to show progress
            **preprocess_params: Parameters for preprocessing
            
        Returns:
            List[str]: Preprocessed documents
        """
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
        
        preprocessed_docs = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(documents, desc="Preprocessing documents")
            except ImportError:
                iterator = documents
                print(f"Processing {len(documents)} documents...")
        else:
            iterator = documents
        
        for doc in iterator:
            if pd.isna(doc) or not isinstance(doc, str):
                preprocessed_docs.append("")
            else:
                preprocessed_docs.append(self.preprocess_text(doc, **preprocess_params))
        
        return preprocessed_docs
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Detected language code
        """
        if not TEXTBLOB_AVAILABLE:
            return 'unknown'
        
        try:
            return TextBlob(text).detect_language()
        except:
            return 'unknown'
    
    def filter_by_language(self, documents: List[str], 
                          target_language: str = 'en') -> Tuple[List[str], List[int]]:
        """
        Filter documents by language.
        
        Args:
            documents (List[str]): List of documents
            target_language (str): Target language code (e.g., 'en' for English)
            
        Returns:
            Tuple[List[str], List[int]]: Filtered documents and their original indices
        """
        if not TEXTBLOB_AVAILABLE:
            print("TextBlob not available. Returning all documents.")
            return documents, list(range(len(documents)))
        
        filtered_docs = []
        filtered_indices = []
        
        for i, doc in enumerate(documents):
            if self.detect_language(doc) == target_language:
                filtered_docs.append(doc)
                filtered_indices.append(i)
        
        return filtered_docs, filtered_indices
    
    def get_text_statistics(self, documents: List[str]) -> Dict:
        """
        Calculate comprehensive statistics for a document collection.
        
        Args:
            documents (List[str]): List of documents
            
        Returns:
            Dict: Statistics dictionary
        """
        stats = {
            'total_documents': len(documents),
            'total_words': 0,
            'total_characters': 0,
            'avg_doc_length': 0,
            'avg_sentence_length': 0,
            'vocabulary_size': 0,
            'most_common_words': [],
            'document_lengths': [],
            'empty_documents': 0
        }
        
        all_words = []
        doc_lengths = []
        total_sentences = 0
        
        for doc in documents:
            if not doc or pd.isna(doc):
                stats['empty_documents'] += 1
                doc_lengths.append(0)
                continue
            
            # Basic statistics
            words = doc.split()
            doc_length = len(words)
            doc_lengths.append(doc_length)
            all_words.extend(words)
            
            stats['total_characters'] += len(doc)
            
            # Sentence count (basic approximation)
            sentences = doc.count('.') + doc.count('!') + doc.count('?')
            total_sentences += max(1, sentences)  # At least 1 sentence per doc
        
        stats['total_words'] = len(all_words)
        stats['document_lengths'] = doc_lengths
        
        if documents:
            stats['avg_doc_length'] = stats['total_words'] / len(documents)
            stats['avg_sentence_length'] = stats['total_words'] / max(1, total_sentences)
        
        # Vocabulary statistics
        if all_words:
            from collections import Counter
            word_counts = Counter(all_words)
            stats['vocabulary_size'] = len(word_counts)
            stats['most_common_words'] = word_counts.most_common(20)
        
        return stats
    
    def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Tuple[str, str]]: List of (entity, type) tuples
        """
        if not NLTK_AVAILABLE:
            return []
        
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)
            
            entities = []
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity = ' '.join([token for token, pos in chunk.leaves()])
                    entity_type = chunk.label()
                    entities.append((entity, entity_type))
            
            return entities
        except:
            return []
    
    def create_custom_stopwords(self, documents: List[str], 
                               top_n: int = 50,
                               min_doc_freq: float = 0.1) -> List[str]:
        """
        Create custom stopwords based on document frequency.
        
        Args:
            documents (List[str]): List of documents
            top_n (int): Number of top frequent words to consider
            min_doc_freq (float): Minimum document frequency (0-1)
            
        Returns:
            List[str]: Custom stopwords
        """
        from collections import Counter
        
        # Count word frequencies across documents
        doc_word_counts = Counter()
        total_docs = len(documents)
        
        for doc in documents:
            if not doc:
                continue
            words = set(doc.lower().split())  # Use set to count document frequency
            doc_word_counts.update(words)
        
        # Filter by document frequency
        custom_stopwords = []
        for word, count in doc_word_counts.most_common(top_n):
            doc_freq = count / total_docs
            if doc_freq >= min_doc_freq:
                custom_stopwords.append(word)
        
        return custom_stopwords
    
    def validate_preprocessing(self, original_docs: List[str], 
                             processed_docs: List[str]) -> Dict:
        """
        Validate preprocessing results.
        
        Args:
            original_docs (List[str]): Original documents
            processed_docs (List[str]): Processed documents
            
        Returns:
            Dict: Validation results
        """
        original_stats = self.get_text_statistics(original_docs)
        processed_stats = self.get_text_statistics(processed_docs)
        
        validation = {
            'document_count_match': len(original_docs) == len(processed_docs),
            'original_total_words': original_stats['total_words'],
            'processed_total_words': processed_stats['total_words'],
            'word_reduction_ratio': 1 - (processed_stats['total_words'] / max(1, original_stats['total_words'])),
            'original_vocab_size': original_stats['vocabulary_size'],
            'processed_vocab_size': processed_stats['vocabulary_size'],
            'vocab_reduction_ratio': 1 - (processed_stats['vocabulary_size'] / max(1, original_stats['vocabulary_size'])),
            'empty_docs_created': processed_stats['empty_documents'] - original_stats['empty_documents']
        }
        
        return validation


class DatasetPreprocessor:
    """
    Specialized preprocessor for different types of datasets.
    """
    
    def __init__(self):
        """Initialize the dataset preprocessor."""
        self.text_processor = TextPreprocessor()
    
    def preprocess_news_articles(self, articles: List[str]) -> List[str]:
        """
        Preprocess news articles with domain-specific settings.
        
        Args:
            articles (List[str]): List of news articles
            
        Returns:
            List[str]: Preprocessed articles
        """
        # News-specific preprocessing parameters
        clean_params = {
            'remove_urls': True,
            'remove_emails': True,
            'remove_mentions': True,
            'remove_hashtags': False,  # Keep hashtags as they might be relevant
            'remove_numbers': False,   # Keep numbers for dates, statistics
            'remove_punctuation': True,
            'lowercase': True
        }
        
        filter_params = {
            'min_length': 3,
            'max_length': 25,
            'alpha_only': True
        }
        
        # Add news-specific stopwords
        news_stopwords = ['said', 'says', 'according', 'reported', 'reuters', 
                         'associated', 'press', 'news', 'story', 'article']
        
        preprocessed_articles = []
        for article in articles:
            if not article or pd.isna(article):
                preprocessed_articles.append("")
                continue
                
            processed = self.text_processor.preprocess_text(
                article,
                clean_params=clean_params,
                filter_params=filter_params,
                remove_stopwords=True
            )
            
            # Remove news-specific stopwords
            tokens = processed.split()
            tokens = [token for token in tokens if token.lower() not in news_stopwords]
            preprocessed_articles.append(' '.join(tokens))
        
        return preprocessed_articles
    
    def preprocess_questions(self, questions: List[str]) -> List[str]:
        """
        Preprocess questions with domain-specific settings.
        
        Args:
            questions (List[str]): List of questions
            
        Returns:
            List[str]: Preprocessed questions
        """
        # Question-specific preprocessing parameters
        clean_params = {
            'remove_urls': True,
            'remove_emails': True,
            'remove_mentions': True,
            'remove_hashtags': True,
            'remove_numbers': False,   # Keep numbers as they're often important in questions
            'remove_punctuation': True,
            'lowercase': True
        }
        
        filter_params = {
            'min_length': 2,
            'max_length': 20,
            'alpha_only': False  # Allow alphanumeric for technical terms
        }
        
        # Add question-specific stopwords
        question_stopwords = ['how', 'what', 'why', 'where', 'when', 'who', 'which',
                             'does', 'do', 'can', 'could', 'would', 'should', 'will']
        
        preprocessed_questions = []
        for question in questions:
            if not question or pd.isna(question):
                preprocessed_questions.append("")
                continue
                
            processed = self.text_processor.preprocess_text(
                question,
                clean_params=clean_params,
                filter_params=filter_params,
                remove_stopwords=True
            )
            
            # Keep question words for context but filter out common ones
            tokens = processed.split()
            # Only remove very common question words that don't add meaning
            common_question_words = ['how', 'what', 'does', 'do', 'can']
            tokens = [token for token in tokens if token.lower() not in common_question_words]
            preprocessed_questions.append(' '.join(tokens))
        
        return preprocessed_questions
    
    def preprocess_social_media(self, posts: List[str]) -> List[str]:
        """
        Preprocess social media posts with domain-specific settings.
        
        Args:
            posts (List[str]): List of social media posts
            
        Returns:
            List[str]: Preprocessed posts
        """
        # Social media specific preprocessing
        clean_params = {
            'remove_urls': True,
            'remove_emails': True,
            'remove_mentions': False,  # Mentions might be relevant
            'remove_hashtags': False,  # Hashtags are important in social media
            'remove_numbers': True,
            'remove_punctuation': True,
            'lowercase': True
        }
        
        filter_params = {
            'min_length': 2,
            'max_length': 15,
            'alpha_only': False
        }
        
        return self.text_processor.preprocess_documents(
            posts,
            clean_params=clean_params,
            filter_params=filter_params,
            remove_stopwords=True,
            use_lemmatization=True
        )


def create_preprocessing_demo():
    """Create a demonstration of preprocessing capabilities."""
    print("🔧 Text Preprocessing Demo")
    print("=" * 40)
    
    # Sample texts
    sample_texts = [
        "Check out this amazing article! https://example.com #AI #MachineLearning @user123",
        "The president announced new healthcare policies affecting millions of Americans today.",
        "How can I improve my machine learning skills? What are the best resources for beginners?",
        "RT @news: Breaking news! Stock market hits record high. $AAPL up 5% 📈"
    ]
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    print("📝 Original texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    print("\n🧹 Preprocessed texts:")
    for i, text in enumerate(sample_texts, 1):
        processed = preprocessor.preprocess_text(text)
        print(f"{i}. {processed}")
    
    # Statistics
    print("\n📊 Text Statistics:")
    stats = preprocessor.get_text_statistics(sample_texts)
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total words: {stats['total_words']}")
    print(f"Vocabulary size: {stats['vocabulary_size']}")
    print(f"Average document length: {stats['avg_doc_length']:.1f} words")
    
    # Validation
    processed_texts = [preprocessor.preprocess_text(text) for text in sample_texts]
    validation = preprocessor.validate_preprocessing(sample_texts, processed_texts)
    print(f"\n✅ Validation Results:")
    print(f"Word reduction: {validation['word_reduction_ratio']:.1%}")
    print(f"Vocabulary reduction: {validation['vocab_reduction_ratio']:.1%}")
    
    print("\n🎯 Preprocessing demo completed!")


if __name__ == "__main__":
    create_preprocessing_demo()