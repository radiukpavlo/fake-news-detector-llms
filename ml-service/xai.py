
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
from lime.lime_text import LimeTextExplainer
from captum.attr import IntegratedGradients

@dataclass
class TokenAttribution:
    token: str
    score: float

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def shap_explain(text: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, top_k: int = 10) -> Dict:
    model.eval(); model.to(_device())
    def predict_proba(texts: List[str]):
        with torch.no_grad():
            toks = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(_device())
            probs = torch.softmax(model(**toks).logits, dim=-1).detach().cpu().numpy()
        return probs
    masker = shap.maskers.Text(tokenizer, collapse_mask_token=True)
    explainer = shap.Explainer(predict_proba, masker)
    sv = explainer([text])
    probs = predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))
    tokens = sv.data[0]
    scores = sv.values[0][pred_idx]
    pairs = [TokenAttribution(token=tokens[i], score=float(scores[i])) for i in range(len(tokens))]
    pairs = sorted(pairs, key=lambda p: abs(p.score), reverse=True)[:top_k]
    return {"method":"shap","predicted_class_index": pred_idx, "tokens":[p.token for p in pairs], "scores":[p.score for p in pairs]}

def lime_explain(text: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, top_k: int = 10) -> Dict:
    model.eval(); model.to(_device())
    class_names = ["fake","real"]
    def predict_proba(texts: List[str]):
        with torch.no_grad():
            toks = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(_device())
            probs = torch.softmax(model(**toks).logits, dim=-1).detach().cpu().numpy()
        return probs
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_proba, num_features=top_k)
    items = exp.as_list()
    return {"method":"lime","tokens":[w for w,_ in items], "scores":[float(s) for _,s in items], "predicted_class": class_names[int(np.argmax(predict_proba([text])[0]))]}

def integrated_gradients_explain(text: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, top_k: int = 10) -> Dict:
    model.eval(); model.to(_device())
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(_device())
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    pred_idx = int(torch.argmax(probs, dim=-1).item())

    emb = model.get_input_embeddings()(enc["input_ids"])
    baseline = torch.zeros_like(emb)
    ig = IntegratedGradients(lambda inputs_embeds, attention_mask: model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits[:, pred_idx])
    attrs = ig.attribute(emb, baselines=baseline, additional_forward_args=(enc["attention_mask"],), n_steps=32)
    token_scores = attrs.norm(p=2, dim=-1).detach().cpu().numpy()[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    pairs = sorted([TokenAttribution(t, float(s)) for t,s in zip(tokens, token_scores)], key=lambda p: abs(p.score), reverse=True)[:top_k]
    return {"method":"integrated_gradients","predicted_class_index": pred_idx, "tokens":[p.token for p in pairs], "scores":[p.score for p in pairs]}
