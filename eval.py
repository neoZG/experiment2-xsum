# Loads a fine-tuned checkpoint and runs summarization inference using a Hugging Face pipeline 
# with fixed beam-search settings. It processes the test split (all languages or specific subset), 
# generates summaries, and computes ROUGE metrics using Hugging Face's evaluate library to assess quality.
# (Analogous to HF's summarization evaluation patterns)

import yaml, argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import pipeline
from evaluate import load as load_eval
from utils import model_type_for

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--dataset", choices=["xsum", "xlsum"], required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--beam", type=int, default=4)
parser.add_argument("--max_gen_len", type=int, default=60)
parser.add_argument("--languages", default=None)
args = parser.parse_args()

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
for ex in ds:
    out = pipe(ex["document" if args.dataset=="xsum" else "article"])[0]["summary_text"]
    refs.append(ex["summary"])
    hyps.append(out)

scores = evaluator.compute(predictions=hyps, references=refs)
print({k: round(v*100, 2) for k,v in scores.items()})
