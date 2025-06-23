#!/usr/bin/env python3
"""
Interaction Pattern Analysis for Sequence Recommenders

This script analyzes how interaction patterns evolve across simulation iterations
to help students understand the data characteristics for sequence recommender development.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame, Window

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("InteractionAnalysis") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Import modules
from data_generator import CompetitionDataGenerator
from simulator import CompetitionSimulator
from sample_recommenders import RandomRecommender
from config import DEFAULT_CONFIG


def analyze_interaction_patterns():
    """
    Analyze interaction patterns across simulation iterations to understand
    data characteristics for sequence recommender development.
    """
    print("=== Interaction Pattern Analysis for Sequence Recommenders ===\n")
    
    # Use the same config as the visualization script
    config = DEFAULT_CONFIG.copy()
    config['data_generation']['n_users'] = 1000
    config['data_generation']['n_items'] = 200
    config['data_generation']['seed'] = 42
    
    # Test different initial history densities
    densities_to_test = [0.001, 0.005, 0.01, 0.02]  # Current default is 0.001
    
    print(f"Testing initial history densities: {densities_to_test}")
    print(f"With {config['data_generation']['n_users']} users and {config['data_generation']['n_items']} items")
    print("=" * 60)
    
    results_by_density = {}
    
    for density in densities_to_test:
        print(f"\n📊 ANALYZING DENSITY: {density}")
        print("-" * 40)
        
        # Initialize data generator with current density
        config['data_generation']['initial_history_density'] = density
        data_generator = CompetitionDataGenerator(
            spark_session=spark,
            **config['data_generation']
        )
        
        # Generate data
        users_df = data_generator.generate_users()
        items_df = data_generator.generate_items()
        history_df = data_generator.generate_initial_history(density)
        
        print(f"Initial interactions generated: {history_df.count()}")
        
        # Analyze initial interaction patterns
        initial_stats = analyze_interaction_stats(history_df, iteration_name="Initial")
        
        # Set up simulator to run a few iterations
        user_generator, item_generator = data_generator.setup_data_generators()
        
        simulator_data_dir = f"interaction_analysis_density_{density}"
        if os.path.exists(simulator_data_dir):
            shutil.rmtree(simulator_data_dir)
        
        simulator = CompetitionSimulator(
            user_generator=user_generator,
            item_generator=item_generator,
            data_dir=simulator_data_dir,
            log_df=history_df,
            conversion_noise_mean=config['simulation']['conversion_noise_mean'],
            conversion_noise_std=config['simulation']['conversion_noise_std'],
            spark_session=spark,
            seed=config['data_generation']['seed']
        )
        
        # Use a simple random recommender for analysis
        recommender = RandomRecommender(seed=42)
        recommender.fit(log=history_df, user_features=users_df, item_features=items_df)
        
        # Run 5 iterations and track interaction growth
        iteration_stats = [initial_stats]
        
        for i in range(5):
            print(f"  Running iteration {i+1}...")
            
            # Run one iteration
            metrics, revenue, true_responses = simulator.run_iteration(
                recommender=recommender,
                user_frac=config['simulation']['user_fraction'],
                k=config['simulation']['k'],
                filter_seen_items=config['simulation']['filter_seen_items'],
                iteration=f"iter_{i}"
            )
            
            # Analyze current log state
            current_log = simulator.simulator.log
            stats = analyze_interaction_stats(current_log, iteration_name=f"After Iter {i+1}")
            iteration_stats.append(stats)
        
        results_by_density[density] = iteration_stats
        
        # Clean up
        if os.path.exists(simulator_data_dir):
            shutil.rmtree(simulator_data_dir)
    
    # Generate comprehensive visualization
    visualize_interaction_evolution(results_by_density, densities_to_test)
    
    # Generate recommendations
    generate_recommendations(results_by_density, densities_to_test)
    
    return results_by_density


def analyze_interaction_stats(log_df, iteration_name=""):
    """
    Analyze detailed interaction statistics for a given log.
    
    Args:
        log_df: Interaction log DataFrame
        iteration_name: Name/label for this iteration
        
    Returns:
        dict: Statistics about the interactions
    """
    if log_df is None or log_df.count() == 0:
        return {
            'iteration': iteration_name,
            'total_interactions': 0,
            'unique_users': 0,
            'unique_items': 0,
            'avg_interactions_per_user': 0,
            'max_interactions_per_user': 0,
            'users_with_multiple_interactions': 0,
            'sequence_length_distribution': {}
        }
    
    total_interactions = log_df.count()
    unique_users = log_df.select("user_idx").distinct().count()
    unique_items = log_df.select("item_idx").distinct().count()
    
    # Calculate interactions per user
    interactions_per_user = log_df.groupBy("user_idx").count().toPandas()
    avg_interactions_per_user = interactions_per_user['count'].mean()
    max_interactions_per_user = interactions_per_user['count'].max()
    
    # Users with multiple interactions (sequence length > 1)
    users_with_multiple = (interactions_per_user['count'] > 1).sum()
    
    # Sequence length distribution
    sequence_lengths = interactions_per_user['count'].value_counts().sort_index()
    sequence_length_dist = sequence_lengths.to_dict()
    
    print(f"  {iteration_name}:")
    print(f"    Total interactions: {total_interactions}")
    print(f"    Unique users with interactions: {unique_users}")
    print(f"    Avg interactions per user: {avg_interactions_per_user:.2f}")
    print(f"    Max interactions per user: {max_interactions_per_user}")
    print(f"    Users with multiple interactions: {users_with_multiple} ({users_with_multiple/unique_users*100:.1f}%)")
    
    return {
        'iteration': iteration_name,
        'total_interactions': total_interactions,
        'unique_users': unique_users,
        'unique_items': unique_items,
        'avg_interactions_per_user': avg_interactions_per_user,
        'max_interactions_per_user': max_interactions_per_user,
        'users_with_multiple_interactions': users_with_multiple,
        'pct_users_with_multiple': users_with_multiple/unique_users*100 if unique_users > 0 else 0,
        'sequence_length_distribution': sequence_length_dist
    }


def visualize_interaction_evolution(results_by_density, densities):
    """
    Create visualizations showing how interaction patterns evolve.
    
    Args:
        results_by_density: Results dictionary organized by density
        densities: List of density values tested
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Interaction Pattern Evolution for Sequence Recommenders', fontsize=16)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Total interactions over iterations
    ax = axes[0, 0]
    for i, density in enumerate(densities):
        iterations = [stats['iteration'] for stats in results_by_density[density]]
        total_interactions = [stats['total_interactions'] for stats in results_by_density[density]]
        ax.plot(range(len(iterations)), total_interactions, 'o-', 
                color=colors[i], label=f'Density {density}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Interactions')
    ax.set_title('Total Interactions Growth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Average interactions per user
    ax = axes[0, 1]
    for i, density in enumerate(densities):
        iterations = [stats['iteration'] for stats in results_by_density[density]]
        avg_per_user = [stats['avg_interactions_per_user'] for stats in results_by_density[density]]
        ax.plot(range(len(iterations)), avg_per_user, 'o-', 
                color=colors[i], label=f'Density {density}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Avg Interactions per User')
    ax.set_title('Average Sequence Length Growth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Percentage of users with multiple interactions
    ax = axes[0, 2]
    for i, density in enumerate(densities):
        iterations = [stats['iteration'] for stats in results_by_density[density]]
        pct_multiple = [stats['pct_users_with_multiple'] for stats in results_by_density[density]]
        ax.plot(range(len(iterations)), pct_multiple, 'o-', 
                color=colors[i], label=f'Density {density}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('% Users with Multiple Interactions')
    ax.set_title('Users Available for Sequence Modeling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Sequence length distribution for highest density (final iteration)
    ax = axes[1, 0]
    highest_density = max(densities)
    final_stats = results_by_density[highest_density][-1]
    seq_lengths = list(final_stats['sequence_length_distribution'].keys())
    seq_counts = list(final_stats['sequence_length_distribution'].values())
    
    ax.bar(seq_lengths, seq_counts, alpha=0.7, color=colors[0])
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Number of Users')
    ax.set_title(f'Final Sequence Length Distribution\n(Density {highest_density})')
    ax.grid(True, alpha=0.3)
    
    # 5. Comparison of final states across densities
    ax = axes[1, 1]
    final_avg_interactions = [results_by_density[d][-1]['avg_interactions_per_user'] for d in densities]
    final_pct_multiple = [results_by_density[d][-1]['pct_users_with_multiple'] for d in densities]
    
    x = np.arange(len(densities))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, final_avg_interactions, width, 
                   label='Avg Sequence Length', alpha=0.7, color=colors[0])
    bars2 = ax2.bar(x + width/2, final_pct_multiple, width, 
                    label='% Users w/ Sequences', alpha=0.7, color=colors[1])
    
    ax.set_xlabel('Initial Density')
    ax.set_ylabel('Avg Sequence Length', color=colors[0])
    ax2.set_ylabel('% Users with Sequences', color=colors[1])
    ax.set_title('Final State Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(densities)
    ax.grid(True, alpha=0.3)
    
    # Add combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 6. Recommendation matrix
    ax = axes[1, 2]
    ax.axis('off')
    
    # Create a recommendation table
    recommendations = []
    for density in densities:
        final_stats = results_by_density[density][-1]
        avg_seq = final_stats['avg_interactions_per_user']
        pct_seq = final_stats['pct_users_with_multiple']
        
        if avg_seq >= 2.5 and pct_seq >= 30:
            rating = "✅ Good"
        elif avg_seq >= 2.0 and pct_seq >= 20:
            rating = "⚠️ Moderate"
        else:
            rating = "❌ Poor"
        
        recommendations.append([f"{density}", f"{avg_seq:.2f}", f"{pct_seq:.1f}%", rating])
    
    table_data = [["Density", "Avg Seq Len", "% Multi-Int", "Rating"]] + recommendations
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Sequence Recommender Suitability')
    
    plt.tight_layout()
    plt.savefig('interaction_pattern_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 Visualization saved to 'interaction_pattern_analysis.png'")


def generate_recommendations(results_by_density, densities):
    """
    Generate recommendations for students about sequence recommender development.
    
    Args:
        results_by_density: Results dictionary organized by density
        densities: List of density values tested
    """
    print("\n" + "="*60)
    print("🎯 RECOMMENDATIONS FOR SEQUENCE RECOMMENDER DEVELOPMENT")
    print("="*60)
    
    print("\n📋 ANALYSIS SUMMARY:")
    print("-" * 20)
    
    for density in densities:
        final_stats = results_by_density[density][-1]
        initial_stats = results_by_density[density][0]
        
        print(f"\nDensity {density}:")
        print(f"  Initial avg sequence length: {initial_stats['avg_interactions_per_user']:.2f}")
        print(f"  Final avg sequence length: {final_stats['avg_interactions_per_user']:.2f}")
        print(f"  Final % users with sequences: {final_stats['pct_users_with_multiple']:.1f}%")
        print(f"  Final max sequence length: {final_stats['max_interactions_per_user']}")
    
    print(f"\n🔍 KEY FINDINGS:")
    print("-" * 15)
    
    # Find best density
    best_density = None
    best_score = 0
    
    for density in densities:
        final_stats = results_by_density[density][-1]
        # Score based on average sequence length and percentage with sequences
        score = final_stats['avg_interactions_per_user'] * final_stats['pct_users_with_multiple'] / 100
        if score > best_score:
            best_score = score
            best_density = density
    
    print(f"1. The DEFAULT density (0.001) results in very sparse sequences initially")
    print(f"2. After 5 simulation iterations, interaction density grows significantly") 
    print(f"3. Higher initial density = better sequence data from the start")
    print(f"4. RECOMMENDED density for sequence recommenders: {best_density}")
    
    best_stats = results_by_density[best_density][-1]
    print(f"   → Results in {best_stats['avg_interactions_per_user']:.1f} avg interactions per user")
    print(f"   → {best_stats['pct_users_with_multiple']:.1f}% of users have multiple interactions")
    
    print(f"\n💡 PRACTICAL RECOMMENDATIONS:")
    print("-" * 28)
    
    print("For Sequence Recommender Development:")
    print(f"  1. Use initial_history_density >= {best_density} in generate_initial_history()")
    print("  2. Run at least 3-5 training iterations to build up interaction history")
    print("  3. Focus on users with 3+ interactions for better sequence patterns")
    print("  4. Consider temporal ordering - add timestamp handling to your recommender")
    
    print("\nExample code modification:")
    print("```python")
    print("# In your data generation:")
    print(f"history_df = data_generator.generate_initial_history({best_density})  # Instead of default")
    print("")
    print("# In your sequence recommender:")
    print("def fit(self, log, user_features=None, item_features=None):")
    print("    # Filter to users with multiple interactions")
    print("    user_counts = log.groupBy('user_idx').count()")
    print("    multi_interaction_users = user_counts.filter(sf.col('count') >= 3)")
    print("    sequence_log = log.join(multi_interaction_users, on='user_idx')")
    print("    # Build your sequence model here...")
    print("```")
    
if __name__ == "__main__":
    print("Starting interaction pattern analysis...")
    results = analyze_interaction_patterns()
    print("\nAnalysis complete! Check 'interaction_pattern_analysis.png' for visualizations.") 