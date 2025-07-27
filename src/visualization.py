"""
Topic Modeling Visualization Module

This module provides comprehensive visualization tools for topic modeling results,
including word clouds, topic distributions, and interactive plots.

Author: Ahmad Hammam
GitHub: @Ahmadhammam03
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("WordCloud not available. Install with: pip install wordcloud")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")


class TopicVisualizer:
    """
    Comprehensive visualization toolkit for topic modeling results.
    
    This class provides various visualization methods to help understand
    and interpret topic modeling results including word clouds, distribution
    plots, and interactive visualizations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'whitegrid'):
        """
        Initialize the visualizer.
        
        Args:
            figsize (Tuple[int, int]): Default figure size
            style (str): Seaborn style to use
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_style(style)
        sns.set_palette("husl")
        
        # Color palettes for consistent styling
        self.colors = sns.color_palette("Set2", 20)
        self.topic_colors = sns.color_palette("tab20", 20)
    
    def plot_topic_words(self, topics: Dict, num_words: int = 10, 
                        save_path: Optional[str] = None) -> None:
        """
        Create horizontal bar plots for top words in each topic.
        
        Args:
            topics (Dict): Topics dictionary from TopicModelingPipeline
            num_words (int): Number of top words to display
            save_path (str, optional): Path to save the plot
        """
        n_topics = len(topics)
        cols = 2
        rows = (n_topics + 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        if n_topics == 1:
            axes = np.array([[axes]])
        
        for topic_idx, topic_info in topics.items():
            row = topic_idx // cols
            col = topic_idx % cols
            ax = axes[row, col]
            
            words = topic_info['words'][:num_words]
            weights = topic_info['weights'][:num_words]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(words))
            bars = ax.barh(y_pos, weights, color=self.topic_colors[topic_idx], alpha=0.7)
            
            # Customize the plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words)
            ax.invert_yaxis()
            ax.set_xlabel('Word Weight')
            ax.set_title(f'Topic {topic_idx}: {topic_info["name"]}', 
                        fontweight='bold', fontsize=12)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + max(weights) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # Remove empty subplots
        for i in range(n_topics, rows * cols):
            row = i // cols
            col = i % cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.suptitle('Topic Words Distribution', fontsize=16, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_word_clouds(self, topics: Dict, save_dir: Optional[str] = None) -> None:
        """
        Create word clouds for each topic.
        
        Args:
            topics (Dict): Topics dictionary from TopicModelingPipeline
            save_dir (str, optional): Directory to save word clouds
        """
        if not WORDCLOUD_AVAILABLE:
            print("WordCloud library not available. Please install it to use this feature.")
            return
        
        n_topics = len(topics)
        cols = 3
        rows = (n_topics + 2) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for topic_idx, topic_info in topics.items():
            row = topic_idx // cols
            col = topic_idx % cols
            ax = axes[row, col]
            
            # Create word frequency dictionary
            words = topic_info['words']
            weights = topic_info['weights']
            word_freq = dict(zip(words, weights))
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                colormap='viridis',
                max_words=50,
                relative_scaling=0.5,
                random_state=42
            ).generate_from_frequencies(word_freq)
            
            # Display word cloud
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Topic {topic_idx}: {topic_info["name"]}', 
                        fontweight='bold', fontsize=12)
            ax.axis('off')
            
            # Save individual word cloud if directory provided
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                wordcloud.to_file(f"{save_dir}/topic_{topic_idx}_wordcloud.png")
        
        # Remove empty subplots
        for i in range(n_topics, rows * cols):
            row = i // cols
            col = i % cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.suptitle('Topic Word Clouds', fontsize=16, fontweight='bold', y=1.02)
        plt.show()
    
    def plot_topic_distribution(self, df: pd.DataFrame, topic_column: str = 'Topic',
                               save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of topics across documents.
        
        Args:
            df (pd.DataFrame): DataFrame with topic assignments
            topic_column (str): Name of the topic column
            save_path (str, optional): Path to save the plot
        """
        # Count topics
        topic_counts = df[topic_column].value_counts().sort_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = ax1.bar(topic_counts.index, topic_counts.values, 
                      color=[self.topic_colors[i] for i in topic_counts.index],
                      alpha=0.7)
        ax1.set_xlabel('Topic ID')
        ax1.set_ylabel('Number of Documents')
        ax1.set_title('Topic Distribution (Bar Plot)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(topic_counts.values) * 0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(topic_counts.values, labels=[f'Topic {i}' for i in topic_counts.index],
               colors=[self.topic_colors[i] for i in topic_counts.index],
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('Topic Distribution (Pie Chart)', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_topic_similarity_heatmap(self, model, save_path: Optional[str] = None) -> None:
        """
        Create a heatmap showing topic similarity based on word overlap.
        
        Args:
            model: Trained topic modeling pipeline
            save_path (str, optional): Path to save the plot
        """
        n_topics = len(model.topics)
        similarity_matrix = np.zeros((n_topics, n_topics))
        
        # Calculate Jaccard similarity between topics
        for i in range(n_topics):
            for j in range(n_topics):
                words_i = set(model.topics[i]['words'][:10])
                words_j = set(model.topics[j]['words'][:10])
                
                intersection = len(words_i.intersection(words_j))
                union = len(words_i.union(words_j))
                similarity_matrix[i, j] = intersection / union if union > 0 else 0
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))
        
        sns.heatmap(similarity_matrix, mask=mask, annot=True, cmap='viridis',
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   fmt='.3f')
        
        plt.title('Topic Similarity Heatmap\n(Based on Top 10 Words Overlap)', 
                 fontweight='bold', fontsize=14)
        plt.xlabel('Topic ID')
        plt.ylabel('Topic ID')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_algorithm_comparison(self, lda_model, nmf_model, 
                                 save_path: Optional[str] = None) -> None:
        """
        Create a comparison visualization between LDA and NMF results.
        
        Args:
            lda_model: Trained LDA model
            nmf_model: Trained NMF model
            save_path (str, optional): Path to save the plot
        """
        n_topics = min(len(lda_model.topics), len(nmf_model.topics))
        
        fig, axes = plt.subplots(n_topics, 2, figsize=(16, 4 * n_topics))
        if n_topics == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_topics):
            # LDA words
            lda_words = lda_model.topics[i]['words'][:8]
            lda_weights = lda_model.topics[i]['weights'][:8]
            
            # NMF words
            nmf_words = nmf_model.topics[i]['words'][:8]
            nmf_weights = nmf_model.topics[i]['weights'][:8]
            
            # Plot LDA
            axes[i, 0].barh(range(len(lda_words)), lda_weights, 
                           color='skyblue', alpha=0.7)
            axes[i, 0].set_yticks(range(len(lda_words)))
            axes[i, 0].set_yticklabels(lda_words)
            axes[i, 0].set_title(f'LDA Topic {i}: {lda_model.topics[i]["name"]}')
            axes[i, 0].invert_yaxis()
            
            # Plot NMF
            axes[i, 1].barh(range(len(nmf_words)), nmf_weights, 
                           color='lightcoral', alpha=0.7)
            axes[i, 1].set_yticks(range(len(nmf_words)))
            axes[i, 1].set_yticklabels(nmf_words)
            axes[i, 1].set_title(f'NMF Topic {i}: {nmf_model.topics[i]["name"]}')
            axes[i, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.suptitle('LDA vs NMF Topic Comparison', fontsize=16, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_visualization_demo():
    """Create a demonstration of visualization capabilities."""
    print("🎨 Topic Modeling Visualization Demo")
    print("=" * 40)
    
    # Create sample data
    sample_topics = {
        0: {
            'name': 'Technology & AI',
            'words': ['artificial', 'intelligence', 'machine', 'learning', 'computer', 
                     'data', 'algorithm', 'technology', 'software', 'digital'],
            'weights': [0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
        },
        1: {
            'name': 'Health & Medicine',
            'words': ['health', 'medical', 'doctor', 'patient', 'hospital', 
                     'treatment', 'disease', 'medicine', 'care', 'clinical'],
            'weights': [0.18, 0.14, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
        },
        2: {
            'name': 'Business & Finance',
            'words': ['business', 'company', 'market', 'financial', 'money', 
                     'investment', 'economic', 'profit', 'revenue', 'industry'],
            'weights': [0.16, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
        }
    }
    
    # Create visualizer
    visualizer = TopicVisualizer()
    
    # Demo topic words plot
    print("📊 Creating topic words visualization...")
    visualizer.plot_topic_words(sample_topics, num_words=8)
    
    # Demo topic distribution
    print("📈 Creating topic distribution plot...")
    sample_df = pd.DataFrame({
        'Topic': np.random.choice([0, 1, 2], size=100, p=[0.4, 0.35, 0.25])
    })
    visualizer.plot_topic_distribution(sample_df)
    
    print("✅ Visualization demo completed!")


if __name__ == "__main__":
    create_visualization_demo()