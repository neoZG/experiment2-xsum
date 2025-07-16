# Uses Hugging Face's Trainer API to fine-tune a specified model (encoder–decoder or causal) on the XSUM or XL‑Sum subset. 
# It loads and preprocesses the chosen dataset, applies mixed-precision/memory optimizations, 
# and trains the model for a set number of epochs before saving the checkpoint. 
# Built-in sample limits enable fast iteration using max_train_samples and max_val_samples.
# (Structured similarly to Hugging Face's summarization training examples)

import yaml, argparse
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, Trainer, TrainingArguments
from utils import preprocess, model_type_for

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--model_name", required=True)
parser.add_argument("--dataset", choices=["xsum", "xlsum"], required=True)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--max_train", type=int, default=None)
parser.add_argument("--max_val", type=int, default=None)
parser.add_argument("--output", default="outputs/run")
args = parser.parse_args()

cfg = yaml.safe_load(open("config.yaml"))
ds_cfg = cfg["dataset"][args.dataset]

ds = load_dataset("xsum" if args.dataset=="xsum" else "GEM/xlsum", args.dataset if args.dataset=="xsum" else None)
if args.dataset=="xlsum":
    ds = ds.filter(lambda x: x["language"] in ds_cfg["languages"])
ds = ds["train"] if args.dataset=="xsum" else ds["train"]
if args.max_train: ds = ds.select(range(args.max_train))
val_split = load_dataset("xsum", split="validation") if args.dataset=="xsum" else load_dataset("GEM/xlsum", split="validation")
if args.max_val: val_split = val_split.select(range(args.max_val))

tok = AutoTokenizer.from_pretrained(args.model_name)
model_type = model_type_for(args.model)

# Handle model loading with appropriate dtype
if "bitnet" in args.model_name.lower():
    model = (AutoModelForCausalLM if model_type=="causal" else AutoModelForSeq2SeqLM).from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16
    )
else:
    model = (AutoModelForCausalLM if model_type=="causal" else AutoModelForSeq2SeqLM).from_pretrained(args.model_name)

def tokenize(batch):
    return preprocess(batch, tok, model_type, max_input=512, max_target=64)
train = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
val = val_split.map(tokenize, batched=True, remove_columns=val_split.column_names)

training_args = TrainingArguments(
    output_dir=args.output, per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size, num_train_epochs=args.epochs,
    learning_rate=args.lr, evaluation_strategy="epoch", save_strategy="epoch",
    fp16=cfg["training"]["fp16"],
    gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
    logging_steps=50,
    save_total_limit=2,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train, eval_dataset=val,
                  tokenizer=tok)
trainer.train()
trainer.save_model(args.output)
