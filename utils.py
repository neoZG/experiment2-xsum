import yaml

def model_type_for(model_key):
    """
    Determine if model is causal LM or seq2seq based on config.yaml
    """
    cfg = yaml.safe_load(open("config.yaml"))
    if model_key in cfg["model"]:
        return cfg["model"][model_key]["type"]
    # Handle full model names as well
    for k, v in cfg["model"].items():
        if v["hf_name"] in model_key:
            return v["type"]
    return "seq2seq"  # default

def preprocess(batch, tokenizer, model_type, max_input=512, max_target=64):
    """
    Preprocess batch for training/evaluation
    """
    if model_type == "causal":
        # For causal models, concatenate input and target
        inputs = [f"Summarize: {doc} Summary: {summ}" for doc, summ in 
                 zip(batch["document" if "document" in batch else "article"], batch["summary"])]
        tokenized = tokenizer(inputs, max_length=max_input + max_target, 
                            truncation=True, padding=True, return_tensors="pt")
        # Labels are the same as input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].clone()
    else:
        # For seq2seq models
        inputs = batch["document"] if "document" in batch else batch["article"]
        targets = batch["summary"]
        
        model_inputs = tokenizer(inputs, max_length=max_input, truncation=True, padding=True)
        labels = tokenizer(text_target=targets, max_length=max_target, truncation=True, padding=True)
        
        model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs 