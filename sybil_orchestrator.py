"""
SYBIL — Orchestratore principale v0.1
Interroga le AI, registra le risposte, aggiorna gli score.
Eseguito ogni giorno da GitHub Actions.
"""

import os
import json
import requests
from datetime import datetime, timezone
from supabase import create_client, Client

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://mkmihvdmenjjaspoqowy.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── PROMPT STANDARD ─────────────────────────────────────────────────────────
def build_prompt(question_text: str, question_type: str) -> str:
    format_map = {
        "binary": "SÌ o NO (scrivi solo SÌ oppure NO)",
        "numeric": "un numero (scrivi solo il numero, senza unità di misura)",
        "percentage": "una percentuale (scrivi solo il numero senza il simbolo %)"
    }
    format_instruction = format_map.get(question_type, "una risposta breve")
    return f"""Sei partecipante al Campionato Sybil di accuratezza predittiva.

Domanda: {question_text}

Formato risposta obbligatorio: {format_instruction}

Regola: Rispondi SOLO nel formato richiesto. Nessun disclaimer, nessuna spiegazione, nessun ragionamento.
Nota: Se non rispondi nel formato richiesto, verrà registrato RIFIUTO = 0 punti."""


# ─── INTERROGATORI PER MODELLO ────────────────────────────────────────────────
def ask_gemini(prompt: str, model_string: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY non impostata")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_string}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def ask_anthropic(prompt: str, model_string: str) -> str:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY non impostata")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": model_string,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": prompt}]
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()


def ask_openai(prompt: str, model_string: str) -> str:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY non impostata")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_string,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": prompt}]
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def ask_groq(prompt: str, model_string: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY non impostata")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_string,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": prompt}]
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ─── ROUTER ──────────────────────────────────────────────────────────────────
def ask_model(model: dict, prompt: str) -> tuple[str, bool]:
    """
    Ritorna (risposta, refused).
    refused=True se il modello non ha risposto nel formato atteso.
    """
    provider = model["provider"].lower()
    model_string = model["model_string"]
    try:
        if provider == "google":
            raw = ask_gemini(prompt, model_string)
        elif provider == "anthropic":
            raw = ask_anthropic(prompt, model_string)
        elif provider == "openai":
            raw = ask_openai(prompt, model_string)
        elif provider == "groq":
            raw = ask_groq(prompt, model_string)
        else:
            return "PROVIDER_SCONOSCIUTO", True

        # Risposta troppo lunga = probabile disclaimer = rifiuto
        if len(raw) > 100:
            print(f"  ⚠️  {model['name']}: risposta troppo lunga ({len(raw)} chars), registrato come rifiuto")
            return raw[:200], True

        return raw, False

    except Exception as e:
        print(f"  ❌ {model['name']}: errore — {e}")
        return f"ERRORE: {str(e)}", True


# ─── VERIFICA ESITI ──────────────────────────────────────────────────────────
def verify_meteo_question(question: dict) -> str | None:
    """
    Verifica automatica domande meteo tramite Open-Meteo API.
    Ritorna il valore corretto come stringa, o None se non verificabile.
    """
    # Coordinate Milano (default per ora)
    lat, lon = 45.4642, 9.1900
    yesterday = (datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max"
        f"&start_date={yesterday}&end_date={yesterday}"
        f"&timezone=Europe/Rome"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        temp_max = data["daily"]["temperature_2m_max"][0]
        return str(round(temp_max, 1))
    except Exception as e:
        print(f"  ⚠️  Verifica meteo fallita: {e}")
        return None


# ─── CALCOLO SCORE ────────────────────────────────────────────────────────────
def update_scores(model_id: str, category: str, correct: bool, refused: bool):
    """Aggiorna lo score di un modello per una categoria."""
    # Recupera score attuale
    result = supabase.table("scores").select("*").eq("model_id", model_id).eq("category", category).execute()

    if result.data:
        row = result.data[0]
        new_total = row["total_questions"] + 1
        new_correct = row["correct"] + (1 if correct else 0)
        new_refused = row["refused"] + (1 if refused else 0)
        new_score = round((new_correct / new_total) * 100, 2) if new_total > 0 else 0
        supabase.table("scores").update({
            "total_questions": new_total,
            "correct": new_correct,
            "refused": new_refused,
            "score": new_score,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("model_id", model_id).eq("category", category).execute()
    else:
        # Prima entry per questo modello/categoria
        supabase.table("scores").insert({
            "model_id": model_id,
            "category": category,
            "total_questions": 1,
            "correct": 1 if correct else 0,
            "refused": 1 if refused else 0,
            "score": 100.0 if correct else 0.0
        }).execute()


# ─── PIPELINE PRINCIPALE ─────────────────────────────────────────────────────
def run():
    print(f"\n🔮 SYBIL — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 1. Carica modelli attivi
    models = supabase.table("ai_models").select("*").eq("active", True).execute().data
    print(f"📊 Modelli attivi: {len(models)}")

    # 2. Carica domande aperte (senza risposta ancora, deadline non scaduta)
    now = datetime.now(timezone.utc).isoformat()
    questions = (
        supabase.table("questions")
        .select("*")
        .eq("verified", False)
        .gt("deadline", now)
        .execute().data
    )
    print(f"❓ Domande aperte: {len(questions)}")

    # 3. Per ogni domanda, interroga i modelli che non hanno ancora risposto
    for q in questions:
        print(f"\n📌 [{q['category'].upper()}] {q['text'][:60]}...")
        prompt = build_prompt(q["text"], q["type"])

        for model in models:
            # Controlla se ha già risposto
            existing = (
                supabase.table("responses")
                .select("id")
                .eq("question_id", q["id"])
                .eq("model_id", model["id"])
                .execute().data
            )
            if existing:
                print(f"  ⏭️  {model['name']}: già risposto")
                continue

            print(f"  🤖 {model['name']}...", end=" ")
            response_value, refused = ask_model(model, prompt)
            print(f"→ {'RIFIUTO' if refused else response_value}")

            # Salva risposta
            supabase.table("responses").insert({
                "question_id": q["id"],
                "model_id": model["id"],
                "response_value": response_value,
                "refused": refused,
                "prompt_used": prompt
            }).execute()

    # 4. Verifica esiti scaduti non ancora verificati
    print("\n🔍 Verifica esiti scaduti...")
    expired = (
        supabase.table("questions")
        .select("*")
        .eq("verified", False)
        .lt("deadline", now)
        .execute().data
    )
    print(f"   Domande scadute da verificare: {len(expired)}")

    for q in expired:
        correct_value = None

        # Auto-verifica meteo
        if q["auto_verify"] and q["category"] == "meteo":
            correct_value = verify_meteo_question(q)

        if correct_value:
            # Aggiorna domanda con esito
            supabase.table("questions").update({
                "correct_value": correct_value,
                "verified": True,
                "verified_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", q["id"]).execute()

            print(f"  ✅ [{q['category']}] Esito: {correct_value}")

            # Aggiorna score per ogni modello
            responses = (
                supabase.table("responses")
                .select("*")
                .eq("question_id", q["id"])
                .execute().data
            )
            for resp in responses:
                if resp["refused"]:
                    update_scores(resp["model_id"], q["category"], correct=False, refused=True)
                else:
                    # Confronto semplice per ora (migliorare con tolleranza numerica)
                    is_correct = resp["response_value"].strip().lower() == correct_value.strip().lower()
                    update_scores(resp["model_id"], q["category"], correct=is_correct, refused=False)
        else:
            print(f"  ⏳ [{q['category']}] Verifica manuale necessaria: {q['text'][:50]}...")

    print("\n✅ Pipeline completata.")


if __name__ == "__main__":
    run()
