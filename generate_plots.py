#!/usr/bin/env python3
"""
BitNet Summarization Analysis - Plot Generation Script
Generates comprehensive visualizations for research analysis and thesis presentation.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Suppress warnings and set non-interactive backend
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')  # Non-interactive backend
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create plots directory
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

def load_results():
    """Load all experimental results from JSON files."""
    try:
        # Load comprehensive analysis files
        with open("outputs/experiment_summary/model_comparison.json", "r") as f:
            model_comparison = json.load(f)
        
        with open("outputs/experiment_summary/statistical_analysis.json", "r") as f:
            statistical_analysis = json.load(f)
        
        with open("outputs/experiment_summary/efficiency_analysis.json", "r") as f:
            efficiency_analysis = json.load(f)
        
        # Load individual model results
        individual_results = {}
        model_paths = [
            ("bitnet_xsum", "outputs/bitnet_xsum/results/20240401_143022/eval_summary.json"),
            ("bart_xsum", "outputs/bart_xsum/results/20240401_144015/eval_summary.json"),
            ("gemma_xsum", "outputs/gemma_xsum/results/20240401_145030/eval_summary.json"),
            ("gpt_neox_xsum", "outputs/gpt_neox_xsum/results/20240401_150045/eval_summary.json"),
            ("mbart_xlsum6", "outputs/mbart_xlsum6/results/20240401_151100/eval_summary.json"),
            ("bitnet_xlsum6", "outputs/bitnet_xlsum6/results/20240401_152115/eval_summary.json")
        ]
        
        for model_name, path in model_paths:
            try:
                with open(path, "r") as f:
                    individual_results[model_name] = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Could not find {path}")
        
        return model_comparison, statistical_analysis, efficiency_analysis, individual_results
    
    except Exception as e:
        print(f"Error loading results: {e}")
        return None, None, None, None

def plot_rouge_comparison(model_comparison, save_dir):
    """Plot ROUGE score comparisons across models."""
    # XSUM Models ROUGE Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    xsum_models = model_comparison["models"]["xsum_results"]
    models = list(xsum_models.keys())
    rouge_metrics = ["rouge1", "rouge2", "rougeL"]
    
    for i, metric in enumerate(rouge_metrics):
        scores = [xsum_models[model]["rouge_scores"][metric] for model in models]
        colors = ['#FF6B6B' if model == 'bitnet' else '#4ECDC4' for model in models]
        
        bars = axes[i].bar(models, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        axes[i].set_title(f'{metric.upper()} Scores - English XSUM', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('ROUGE Score', fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / "rouge_comparison_xsum.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Multilingual ROUGE Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    xlsum_models = model_comparison["models"]["xlsum_results"]
    languages = ["en", "es", "hi", "am", "si", "ha"]
    
    for i, lang in enumerate(languages):
        row, col = i // 3, i % 3
        
        mbart_scores = [
            xlsum_models["mbart"]["rouge_scores"]["by_language"][lang]["rouge1"],
            xlsum_models["mbart"]["rouge_scores"]["by_language"][lang]["rouge2"],
            xlsum_models["mbart"]["rouge_scores"]["by_language"][lang]["rougeL"]
        ]
        
        bitnet_scores = [
            xlsum_models["bitnet"]["rouge_scores"]["by_language"][lang]["rouge1"],
            xlsum_models["bitnet"]["rouge_scores"]["by_language"][lang]["rouge2"],
            xlsum_models["bitnet"]["rouge_scores"]["by_language"][lang]["rougeL"]
        ]
        
        x = np.arange(len(rouge_metrics))
        width = 0.35
        
        axes[row, col].bar(x - width/2, mbart_scores, width, label='mBART', color='#4ECDC4', alpha=0.8)
        axes[row, col].bar(x + width/2, bitnet_scores, width, label='BitNet', color='#FF6B6B', alpha=0.8)
        
        axes[row, col].set_title(f'{lang.upper()} - XL-Sum', fontsize=12, fontweight='bold')
        axes[row, col].set_ylabel('ROUGE Score')
        axes[row, col].set_xticks(x)
        axes[row, col].set_xticklabels(['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "rouge_comparison_multilingual.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_vs_efficiency(model_comparison, efficiency_analysis, save_dir):
    """Plot performance vs efficiency trade-offs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare XSUM data
    xsum_models = model_comparison["models"]["xsum_results"]
    efficiency_data = efficiency_analysis["efficiency_analysis"]
    
    model_names = []
    rouge1_scores = []
    memory_usage = []
    inference_speed = []
    energy_consumption = []
    
    for model in xsum_models.keys():
        model_key = f"{model}_xsum"
        model_names.append(model.upper())
        rouge1_scores.append(xsum_models[model]["rouge_scores"]["rouge1"])
        memory_usage.append(efficiency_data["resource_consumption"]["memory_usage"]["inference"][model_key]["peak_gb"])
        inference_speed.append(efficiency_data["scalability_metrics"]["throughput"]["inference_examples_per_second"][model_key])
        energy_consumption.append(efficiency_data["energy_consumption"]["inference_energy_per_1000_examples_wh"][model_key])
    
    colors = ['#FF6B6B' if 'BITNET' in name else '#4ECDC4' for name in model_names]
    
    # Performance vs Memory Usage
    scatter1 = axes[0, 0].scatter(memory_usage, rouge1_scores, c=colors, s=150, alpha=0.8, edgecolors='black')
    axes[0, 0].set_xlabel('Memory Usage (GB)', fontsize=12)
    axes[0, 0].set_ylabel('ROUGE-1 Score', fontsize=12)
    axes[0, 0].set_title('Performance vs Memory Usage', fontsize=14, fontweight='bold')
    for i, name in enumerate(model_names):
        axes[0, 0].annotate(name, (memory_usage[i], rouge1_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Performance vs Inference Speed
    scatter2 = axes[0, 1].scatter(inference_speed, rouge1_scores, c=colors, s=150, alpha=0.8, edgecolors='black')
    axes[0, 1].set_xlabel('Inference Speed (examples/sec)', fontsize=12)
    axes[0, 1].set_ylabel('ROUGE-1 Score', fontsize=12)
    axes[0, 1].set_title('Performance vs Inference Speed', fontsize=14, fontweight='bold')
    for i, name in enumerate(model_names):
        axes[0, 1].annotate(name, (inference_speed[i], rouge1_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Performance vs Energy Consumption
    scatter3 = axes[1, 0].scatter(energy_consumption, rouge1_scores, c=colors, s=150, alpha=0.8, edgecolors='black')
    axes[1, 0].set_xlabel('Energy per 1000 examples (Wh)', fontsize=12)
    axes[1, 0].set_ylabel('ROUGE-1 Score', fontsize=12)
    axes[1, 0].set_title('Performance vs Energy Consumption', fontsize=14, fontweight='bold')
    for i, name in enumerate(model_names):
        axes[1, 0].annotate(name, (energy_consumption[i], rouge1_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Efficiency Spider Chart for BitNet vs BART
    metrics = ['ROUGE-1', 'Memory Eff.', 'Speed', 'Energy Eff.']
    
    # Normalize metrics (0-1 scale, higher is better)
    bitnet_values = [
        rouge1_scores[0] / max(rouge1_scores),  # ROUGE-1 (BitNet is first)
        1 - (memory_usage[0] / max(memory_usage)),  # Memory efficiency (lower usage = higher efficiency)
        inference_speed[0] / max(inference_speed),  # Speed
        1 - (energy_consumption[0] / max(energy_consumption))  # Energy efficiency
    ]
    
    bart_idx = model_names.index('BART')
    bart_values = [
        rouge1_scores[bart_idx] / max(rouge1_scores),
        1 - (memory_usage[bart_idx] / max(memory_usage)),
        inference_speed[bart_idx] / max(inference_speed),
        1 - (energy_consumption[bart_idx] / max(energy_consumption))
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    bitnet_values += bitnet_values[:1]
    bart_values += bart_values[:1]
    
    ax_spider = plt.subplot(2, 2, 4, projection='polar')
    ax_spider.plot(angles, bitnet_values, 'o-', linewidth=2, label='BitNet', color='#FF6B6B')
    ax_spider.fill(angles, bitnet_values, alpha=0.25, color='#FF6B6B')
    ax_spider.plot(angles, bart_values, 'o-', linewidth=2, label='BART', color='#4ECDC4')
    ax_spider.fill(angles, bart_values, alpha=0.25, color='#4ECDC4')
    
    ax_spider.set_xticks(angles[:-1])
    ax_spider.set_xticklabels(metrics)
    ax_spider.set_ylim(0, 1)
    ax_spider.set_title('BitNet vs BART - Multi-dimensional Comparison', fontweight='bold', pad=20)
    ax_spider.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(save_dir / "performance_vs_efficiency.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_language_resource_analysis(model_comparison, statistical_analysis, save_dir):
    """Plot quantization impact across language resource levels."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Language-wise performance comparison
    xlsum_data = model_comparison["models"]["xlsum_results"]
    languages = ["en", "es", "hi", "am", "si", "ha"]
    lang_labels = ["English", "Spanish", "Hindi", "Amharic", "Sinhala", "Hausa"]
    
    mbart_scores = []
    bitnet_scores = []
    performance_gaps = []
    
    for lang in languages:
        mbart_score = xlsum_data["mbart"]["rouge_scores"]["by_language"][lang]["rouge1"]
        bitnet_score = xlsum_data["bitnet"]["rouge_scores"]["by_language"][lang]["rouge1"]
        gap = ((mbart_score - bitnet_score) / mbart_score) * 100
        
        mbart_scores.append(mbart_score)
        bitnet_scores.append(bitnet_score)
        performance_gaps.append(gap)
    
    x = np.arange(len(languages))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, mbart_scores, width, label='mBART', color='#4ECDC4', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, bitnet_scores, width, label='BitNet', color='#FF6B6B', alpha=0.8)
    
    axes[0].set_xlabel('Language', fontsize=12)
    axes[0].set_ylabel('ROUGE-1 Score', fontsize=12)
    axes[0].set_title('ROUGE-1 Performance by Language', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(lang_labels, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.3,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Performance degradation by resource level
    resource_levels = ["High", "High", "Mid", "Low", "Low", "Low"]
    colors_by_resource = ['#2E8B57' if level == 'High' else '#FFD700' if level == 'Mid' else '#DC143C' 
                         for level in resource_levels]
    
    bars3 = axes[1].bar(lang_labels, performance_gaps, color=colors_by_resource, alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Language', fontsize=12)
    axes[1].set_ylabel('Performance Gap (%)', fontsize=12)
    axes[1].set_title('Quantization Impact by Language Resource Level', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, gap in zip(bars3, performance_gaps):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                    f'{gap:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add resource level legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2E8B57', label='High Resource'),
                      Patch(facecolor='#FFD700', label='Mid Resource'),
                      Patch(facecolor='#DC143C', label='Low Resource')]
    axes[1].legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_dir / "language_resource_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_progression(save_dir):
    """Plot training progression for key models."""
    # Load training data
    training_files = [
        ("BitNet XSUM", "outputs/bitnet_xsum/trainer_state.json"),
        ("BART XSUM", "outputs/bart_xsum/trainer_state.json"),
        ("mBART XL-Sum", "outputs/mbart_xlsum6/trainer_state.json"),
        ("BitNet XL-Sum", "outputs/bitnet_xlsum6/trainer_state.json")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, (model_name, file_path) in enumerate(training_files):
        row, col = idx // 2, idx % 2
        
        try:
            with open(file_path, "r") as f:
                trainer_data = json.load(f)
            
            log_history = trainer_data["log_history"]
            
            # Extract training and validation data
            train_steps = []
            train_losses = []
            eval_steps = []
            eval_losses = []
            eval_rouge1 = []
            
            for entry in log_history:
                if "loss" in entry and "eval_loss" not in entry:
                    train_steps.append(entry["step"])
                    train_losses.append(entry["loss"])
                elif "eval_loss" in entry:
                    eval_steps.append(entry["step"])
                    eval_losses.append(entry["eval_loss"])
                    if isinstance(entry.get("eval_rouge1"), (int, float)):
                        eval_rouge1.append(entry["eval_rouge1"])
                    elif isinstance(entry.get("eval_rouge1"), dict):
                        # For multilingual models, use average
                        avg_rouge = np.mean(list(entry["eval_rouge1"].values()))
                        eval_rouge1.append(avg_rouge)
            
            # Plot training loss
            color = '#FF6B6B' if 'BitNet' in model_name else '#4ECDC4'
            axes[row, col].plot(train_steps, train_losses, label='Training Loss', color=color, linewidth=2)
            axes[row, col].plot(eval_steps, eval_losses, label='Validation Loss', color=color, linestyle='--', linewidth=2)
            
            # Add ROUGE-1 on secondary y-axis
            ax2 = axes[row, col].twinx()
            if eval_rouge1:
                ax2.plot(eval_steps, eval_rouge1, label='ROUGE-1', color='#FF8C00', marker='o', linewidth=2)
                ax2.set_ylabel('ROUGE-1 Score', color='#FF8C00', fontsize=12)
                ax2.tick_params(axis='y', labelcolor='#FF8C00')
            
            axes[row, col].set_xlabel('Training Steps', fontsize=12)
            axes[row, col].set_ylabel('Loss', fontsize=12)
            axes[row, col].set_title(f'{model_name} - Training Progression', fontsize=12, fontweight='bold')
            axes[row, col].legend(loc='upper right')
            axes[row, col].grid(True, alpha=0.3)
            
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f'Data not available\n{str(e)}', 
                               ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(f'{model_name} - Training Progression', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_progression.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_statistical_analysis(statistical_analysis, save_dir):
    """Plot statistical significance and confidence intervals."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Effect sizes comparison
    xsum_comparisons = statistical_analysis["statistical_analysis"]["pairwise_comparisons"]["xsum_models"]
    
    comparisons = []
    effect_sizes = []
    p_values = []
    significance = []
    
    for comp_name, comp_data in xsum_comparisons.items():
        if comp_name != "bart_vs_gpt_neox":  # Focus on BitNet comparisons
            comparisons.append(comp_name.replace("_vs_bitnet", " vs BitNet").replace("_", " ").title())
            
            # Map effect size to numeric value
            effect_map = {"small": 0.2, "medium": 0.5, "large": 0.8}
            effect_sizes.append(effect_map.get(comp_data["effect_size"], 0.5))
            p_values.append(comp_data["p_values"]["rouge1"])
            significance.append(comp_data["significant"])
    
    colors = ['#DC143C' if sig else '#32CD32' for sig in significance]
    
    bars = axes[0, 0].barh(comparisons, effect_sizes, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 0].set_xlabel('Effect Size (Cohen\'s d)', fontsize=12)
    axes[0, 0].set_title('Effect Sizes - Model Comparisons vs BitNet', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small Effect')
    axes[0, 0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium Effect')
    axes[0, 0].axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large Effect')
    
    # Add p-value annotations
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        axes[0, 0].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'p={p_val:.3f}', va='center', fontsize=10)
    
    axes[0, 0].legend()
    
    # Confidence intervals plot
    ci_data = statistical_analysis["statistical_analysis"]["confidence_intervals"]["xsum_models"]
    models = list(ci_data.keys())
    rouge1_means = []
    rouge1_lower = []
    rouge1_upper = []
    
    for model in models:
        lower, upper = ci_data[model]["rouge1"]
        mean = (lower + upper) / 2
        rouge1_means.append(mean)
        rouge1_lower.append(mean - lower)
        rouge1_upper.append(upper - mean)
    
    model_colors = ['#FF6B6B' if model == 'bitnet' else '#4ECDC4' for model in models]
    
    axes[0, 1].errorbar(range(len(models)), rouge1_means, 
                       yerr=[rouge1_lower, rouge1_upper], 
                       fmt='o', markersize=8, capsize=5, capthick=2,
                       color='black', ecolor='gray')
    
    for i, (model, color) in enumerate(zip(models, model_colors)):
        axes[0, 1].scatter(i, rouge1_means[i], s=150, color=color, alpha=0.8, edgecolors='black', zorder=5)
    
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels([m.upper() for m in models])
    axes[0, 1].set_ylabel('ROUGE-1 Score', fontsize=12)
    axes[0, 1].set_title('ROUGE-1 Confidence Intervals (95%)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Language-wise significance
    xlsum_by_lang = statistical_analysis["statistical_analysis"]["pairwise_comparisons"]["xlsum_models"]["mbart_vs_bitnet_by_language"]
    
    languages = list(xlsum_by_lang.keys())
    lang_names = ["English", "Spanish", "Hindi", "Amharic", "Sinhala", "Hausa"]
    differences = [xlsum_by_lang[lang]["rouge1_difference"] for lang in languages]
    p_vals = [xlsum_by_lang[lang]["p_value"] for lang in languages]
    significant = [xlsum_by_lang[lang]["significant"] for lang in languages]
    
    colors = ['#32CD32' if not sig else '#DC143C' for sig in significant]
    
    bars = axes[1, 0].bar(lang_names, differences, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 0].set_ylabel('ROUGE-1 Difference', fontsize=12)
    axes[1, 0].set_title('mBART vs BitNet - Language-wise Differences', fontsize=12, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add significance markers
    for i, (bar, p_val, sig) in enumerate(zip(bars, p_vals, significant)):
        marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       marker, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Correlation analysis
    resource_levels = [3, 3, 2, 1, 1, 1]  # High=3, Mid=2, Low=1
    correlation_coef = statistical_analysis["statistical_analysis"]["effect_sizes"]["language_resource_correlation"]["correlation_coefficient"]
    
    axes[1, 1].scatter(resource_levels, differences, s=150, alpha=0.8, 
                      c=colors, edgecolors='black', linewidth=2)
    
    # Add trend line
    z = np.polyfit(resource_levels, differences, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(resource_levels), max(resource_levels), 100)
    axes[1, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    axes[1, 1].set_xlabel('Language Resource Level', fontsize=12)
    axes[1, 1].set_ylabel('Performance Gap (ROUGE-1)', fontsize=12)
    axes[1, 1].set_title(f'Resource Level vs Quantization Impact\n(r = {correlation_coef:.2f})', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks([1, 2, 3])
    axes[1, 1].set_xticklabels(['Low', 'Mid', 'High'])
    
    # Add language labels
    for i, lang_name in enumerate(lang_names):
        axes[1, 1].annotate(lang_name, (resource_levels[i], differences[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / "statistical_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_efficiency_breakdown(efficiency_analysis, save_dir):
    """Plot detailed efficiency analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    efficiency_data = efficiency_analysis["efficiency_analysis"]
    
    # Memory efficiency comparison
    models = ["bitnet", "bart", "gemma", "gpt_neox"]
    model_keys = [f"{model}_xsum" for model in models]
    
    training_memory = [efficiency_data["resource_consumption"]["memory_usage"]["training"][key]["peak_gb"] 
                      for key in model_keys]
    inference_memory = [efficiency_data["resource_consumption"]["memory_usage"]["inference"][key]["peak_gb"] 
                       for key in model_keys]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, training_memory, width, label='Training', alpha=0.8)
    bars2 = axes[0, 0].bar(x + width/2, inference_memory, width, label='Inference', alpha=0.8)
    
    axes[0, 0].set_xlabel('Models', fontsize=12)
    axes[0, 0].set_ylabel('Memory Usage (GB)', fontsize=12)
    axes[0, 0].set_title('Memory Usage Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([m.upper() for m in models])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Energy consumption
    training_energy = [efficiency_data["energy_consumption"]["training_energy_kwh"][key] 
                      for key in model_keys]
    inference_energy = [efficiency_data["energy_consumption"]["inference_energy_per_1000_examples_wh"][key] 
                       for key in model_keys]
    
    bars3 = axes[0, 1].bar(x - width/2, training_energy, width, label='Training (kWh)', alpha=0.8)
    ax_right = axes[0, 1].twinx()
    bars4 = ax_right.bar(x + width/2, inference_energy, width, label='Inference (Wh/1k)', 
                        alpha=0.8, color='orange')
    
    axes[0, 1].set_xlabel('Models', fontsize=12)
    axes[0, 1].set_ylabel('Training Energy (kWh)', fontsize=12, color='blue')
    ax_right.set_ylabel('Inference Energy (Wh/1k examples)', fontsize=12, color='orange')
    axes[0, 1].set_title('Energy Consumption Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([m.upper() for m in models])
    
    # Cost analysis
    training_costs = [efficiency_data["cost_analysis"]["training_cost_usd"]["hardware_hours"][key]["total"] 
                     for key in model_keys]
    inference_costs = [efficiency_data["cost_analysis"]["inference_cost_per_million_examples"][key] 
                      for key in model_keys]
    
    bars5 = axes[0, 2].bar(x - width/2, training_costs, width, label='Training ($)', alpha=0.8)
    ax_right2 = axes[0, 2].twinx()
    bars6 = ax_right2.bar(x + width/2, inference_costs, width, label='Inference ($/M)', 
                         alpha=0.8, color='red')
    
    axes[0, 2].set_xlabel('Models', fontsize=12)
    axes[0, 2].set_ylabel('Training Cost ($)', fontsize=12, color='blue')
    ax_right2.set_ylabel('Inference Cost ($/Million)', fontsize=12, color='red')
    axes[0, 2].set_title('Cost Analysis', fontsize=12, fontweight='bold')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels([m.upper() for m in models])
    
    # BitNet advantages radar chart
    advantages = efficiency_data["bitnet_advantages"]
    metrics = ["Memory vs BART", "Memory vs Gemma", "Memory vs GPT-NeoX", 
              "Energy vs BART", "Energy vs Gemma", "Energy vs GPT-NeoX"]
    values = [advantages["memory_reduction"]["vs_bart"],
             advantages["memory_reduction"]["vs_gemma"],
             advantages["memory_reduction"]["vs_gpt_neox"],
             advantages["energy_reduction"]["vs_bart"],
             advantages["energy_reduction"]["vs_gemma"],
             advantages["energy_reduction"]["vs_gpt_neox"]]
    
    # Convert to percentages
    values = [v * 100 for v in values]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]
    
    ax_radar = plt.subplot(2, 3, 4, projection='polar')
    ax_radar.plot(angles, values, 'o-', linewidth=2, color='#FF6B6B')
    ax_radar.fill(angles, values, alpha=0.25, color='#FF6B6B')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics, fontsize=10)
    ax_radar.set_ylim(0, max(values) * 1.1)
    ax_radar.set_title('BitNet Efficiency Advantages (%)', fontweight='bold', pad=20)
    
    # Throughput comparison
    throughput_training = [efficiency_data["scalability_metrics"]["throughput"]["training_examples_per_hour"][key] 
                          for key in model_keys]
    throughput_inference = [efficiency_data["scalability_metrics"]["throughput"]["inference_examples_per_second"][key] 
                           for key in model_keys]
    
    bars7 = axes[1, 1].bar(x - width/2, throughput_training, width, label='Training (ex/hr)', alpha=0.8)
    ax_right3 = axes[1, 1].twinx()
    bars8 = ax_right3.bar(x + width/2, throughput_inference, width, label='Inference (ex/sec)', 
                         alpha=0.8, color='green')
    
    axes[1, 1].set_xlabel('Models', fontsize=12)
    axes[1, 1].set_ylabel('Training Throughput (ex/hr)', fontsize=12, color='blue')
    ax_right3.set_ylabel('Inference Throughput (ex/sec)', fontsize=12, color='green')
    axes[1, 1].set_title('Throughput Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([m.upper() for m in models])
    
    # Latency analysis
    latency_avg = [efficiency_data["scalability_metrics"]["latency"]["inference_ms_per_example"][key] 
                  for key in model_keys]
    latency_p95 = [efficiency_data["scalability_metrics"]["latency"]["p95_latency_ms"][key] 
                  for key in model_keys]
    
    bars9 = axes[1, 2].bar(x - width/2, latency_avg, width, label='Average', alpha=0.8)
    bars10 = axes[1, 2].bar(x + width/2, latency_p95, width, label='P95', alpha=0.8)
    
    axes[1, 2].set_xlabel('Models', fontsize=12)
    axes[1, 2].set_ylabel('Latency (ms)', fontsize=12)
    axes[1, 2].set_title('Inference Latency Comparison', fontsize=12, fontweight='bold')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels([m.upper() for m in models])
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "efficiency_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all plots for the BitNet investigation."""
    print("🎨 Starting plot generation for BitNet investigation...")
    
    # Load all data
    model_comparison, statistical_analysis, efficiency_analysis, individual_results = load_results()
    
    if not all([model_comparison, statistical_analysis, efficiency_analysis]):
        print("❌ Error: Could not load required data files")
        return
    
    print("✅ Data loaded successfully")
    
    # Generate all plots
    print("📊 Generating ROUGE comparison plots...")
    plot_rouge_comparison(model_comparison, plots_dir)
    
    print("⚡ Generating performance vs efficiency plots...")
    plot_performance_vs_efficiency(model_comparison, efficiency_analysis, plots_dir)
    
    print("🌍 Generating language resource analysis plots...")
    plot_language_resource_analysis(model_comparison, statistical_analysis, plots_dir)
    
    print("📈 Generating training progression plots...")
    plot_training_progression(plots_dir)
    
    print("📊 Generating statistical analysis plots...")
    plot_statistical_analysis(statistical_analysis, plots_dir)
    
    print("💰 Generating efficiency breakdown plots...")
    plot_efficiency_breakdown(efficiency_analysis, plots_dir)
    
    print(f"""
🎉 Plot generation complete! 

Generated plots:
📁 {plots_dir}/
  ├── rouge_comparison_xsum.png
  ├── rouge_comparison_multilingual.png
  ├── performance_vs_efficiency.png
  ├── language_resource_analysis.png
  ├── training_progression.png
  ├── statistical_analysis.png
  └── efficiency_breakdown.png

These plots are ready for your thesis and research presentation!
    """)

if __name__ == "__main__":
    main() 