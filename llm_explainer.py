# src/llm_explainer.py
from typing import Dict

# Lazy-load so Streamlit startup stays fast if transformers/torch aren't present
_HAS_LLM = False
_tok = None
_mdl = None

def _lazy_load():
    """Try to load a small local LLM (FLAN-T5). If it fails, keep fallback mode."""
    global _HAS_LLM, _tok, _mdl
    if _tok is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            _tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
            _mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
            _HAS_LLM = True
        except Exception:
            _HAS_LLM = False

TEMPLATE = (
    "You are a banking fraud analyst. Explain in 2-4 short sentences for a non-technical user "
    "why this transaction is {label}. Keep it simple.\n"
    "Inputs:\n"
    "- amount: {amount}\n"
    "- hour: {hour}\n"
    "- distance_km: {distance}\n"
    "- minutes_since_last: {minutes}\n"
    "- velocity: {velocity}\n"
    "- auth_method_code: {auth}\n"
    "- merchant_category_code: {mcc}\n"
    "- foreign_currency: {fx}\n"
    "- anomaly_signals: {signals}\n"
)

def _signals_from_row(row: Dict) -> str:
    """Lightweight heuristics to surface human-friendly reasons even without an LLM."""
    s = []
    try:
        if float(row.get("is_foreign_currency", 0)) == 1:
            s.append("foreign currency usage")
        if float(row.get("distance_time_ratio", 0) or 0) > 50:
            s.append("long distance in short time")
        if float(row.get("transaction_velocity", 0) or 0) >= 8:
            s.append("unusually high transaction frequency")
        if int(row.get("hour", -1)) in [0, 1, 2, 3, 4]:
            s.append("night-time transaction")
        amt = float(row.get("transaction_amount", 0) or 0)
        if amt > 0 and "p99_amount" in row and row["p99_amount"] is not None:
            try:
                if amt >= float(row["p99_amount"]):
                    s.append("amount higher than 99th percentile")
            except Exception:
                pass
    except Exception:
        pass
    if not s:
        s.append("pattern deviates from typical behavior")
    return ", ".join(s)

def llm_or_fallback_explain(row: Dict, label: str = "FRAUD") -> str:
    """Return a short, user-friendly explanation. Uses local FLAN-T5 if available; otherwise, a rule-based message."""
    _lazy_load()
    signals = _signals_from_row(row)
    if _HAS_LLM:
        prompt = TEMPLATE.format(
            label=label,
            amount=row.get("transaction_amount"),
            hour=row.get("hour"),
            distance=row.get("distance_between_transactions_km"),
            minutes=row.get("time_since_last_transaction_min"),
            velocity=row.get("transaction_velocity"),
            auth=row.get("authentication_method"),
            mcc=row.get("transaction_category"),
            fx=row.get("is_foreign_currency"),
            signals=signals,
        )
        inputs = _tok(prompt, return_tensors="pt")
        out = _mdl.generate(**inputs, max_new_tokens=90)
        return _tok.decode(out[0], skip_special_tokens=True)
    # Fallback text if transformers/torch or the model is unavailable
    return f"Flagged as {label} due to {signals}."
