# Loads a fine-tuned checkpoint and runs summarization inference using a Hugging Face pipeline 
# with fixed beam-search settings. It processes the test split (all languages or specific subset), 
# generates summaries, and computes ROUGE metrics using Hugging Face's evaluate library to assess quality.
# (Analogous to HF's summarization evaluation patterns)

import yaml, argparse
import torch
import random
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import pipeline
from evaluate import load as load_eval
from utils import model_type_for

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def get_gpu_memory():
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / (1024**3),  # GB
            "reserved": torch.cuda.memory_reserved() / (1024**3)     # GB
        }
    return {"allocated": 0, "reserved": 0}

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--dataset", choices=["xsum", "xlsum"], required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--beam", type=int, default=4)
parser.add_argument("--max_gen_len", type=int, default=60)
parser.add_argument("--languages", default=None)
parser.add_argument("--output_dir", default=None, help="Directory to save predictions and metrics")
args = parser.parse_args()

# Setup output directory
if args.output_dir is None:
    args.output_dir = Path(args.model) / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(open("config.yaml"))
ds_cfg = cfg["dataset"][args.dataset]
ds = (load_dataset("xsum", split="test") if args.dataset=="xsum"
     else load_dataset("GEM/xlsum", split="test").filter(lambda x: x["language"] in args.languages.split(",")))
tok = AutoTokenizer.from_pretrained(args.model)
model_type = None
for k,v in cfg["model"].items():
    if v["hf_name"] in args.model: model_type = v["type"]

# Handle model loading with appropriate dtype
if "bitnet" in args.model.lower():
    model = (AutoModelForCausalLM if model_type=="causal" else AutoModelForSeq2SeqLM).from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16
    )
else:
    model = (AutoModelForCausalLM if model_type=="causal" else AutoModelForSeq2SeqLM).from_pretrained(args.model)

pipe = pipeline("summarization", model=model, tokenizer=tok,
                framework="pt", device=0,
                batch_size=args.batch_size,
                max_length=args.max_gen_len,
                num_beams=args.beam)

evaluator = load_eval("rouge")
refs, hyps = [], []
predictions = []
total_time = 0
total_examples = len(ds)
memory_stats = []

print(f"Starting evaluation on {total_examples} examples...")

try:
    for i, ex in enumerate(ds):
        start_time = time.time()
        
        try:
            out = pipe(ex["document" if args.dataset=="xsum" else "article"])[0]["summary_text"]
        except Exception as e:
            print(f"Error generating summary for example {i}: {str(e)}")
            continue
            
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time
        
        refs.append(ex["summary"])
        hyps.append(out)
        
        # Save detailed prediction info
        pred_info = {
            "id": i,
            "document": ex["document" if args.dataset=="xsum" else "article"],
            "reference": ex["summary"],
            "prediction": out,
            "language": ex.get("language", "en"),
            "inference_time": inference_time
        }
        predictions.append(pred_info)
        
        # Track memory every 100 examples
        if i % 100 == 0:
            memory_stats.append({
                "example": i,
                "memory": get_gpu_memory()
            })
            
        if i % 100 == 0:
            print(f"Processed {i}/{total_examples} examples...")

except Exception as e:
    print(f"Evaluation interrupted due to error: {str(e)}")
    
# Calculate metrics
scores = evaluator.compute(predictions=hyps, references=refs)
rouge_scores = {k: round(v*100, 2) for k,v in scores.items()}

# Prepare evaluation summary
eval_summary = {
    "dataset": args.dataset,
    "model": args.model,
    "total_examples": total_examples,
    "completed_examples": len(predictions),
    "rouge_scores": rouge_scores,
    "performance_metrics": {
        "total_time_seconds": total_time,
        "average_time_per_example": total_time / len(predictions) if predictions else 0,
        "examples_per_second": len(predictions) / total_time if total_time > 0 else 0
    },
    "memory_tracking": memory_stats,
    "generation_config": {
        "batch_size": args.batch_size,
        "beam_size": args.beam,
        "max_length": args.max_gen_len
    }
}

# Save results
print("\nSaving results...")
with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
    
with open(output_dir / "eval_summary.json", "w", encoding="utf-8") as f:
    json.dump(eval_summary, f, ensure_ascii=False, indent=2)

print("\nEvaluation Results:")
print(f"ROUGE Scores: {rouge_scores}")
print(f"\nPerformance Metrics:")
print(f"Total Time: {total_time:.2f} seconds")
print(f"Average Time per Example: {eval_summary['performance_metrics']['average_time_per_example']:.3f} seconds")
print(f"Examples per Second: {eval_summary['performance_metrics']['examples_per_second']:.2f}")
print(f"\nResults saved to: {output_dir}")
