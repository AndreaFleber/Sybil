"""
SYBIL — Orchestratore principale v0.2
Interroga le AI, registra le risposte, verifica esiti, aggiorna score.
Eseguito ogni giorno da GitHub Actions.
"""

import os
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
def ask_model(model: dict, prompt: str) -> tuple:
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

        if len(raw) > 100:
            print(f"  ⚠️  {model['name']}: risposta troppo lunga, registrato come rifiuto")
            return raw[:200], True

        return raw, False

    except Exception as e:
        print(f"  ❌ {model['name']}: errore — {e}")
        return f"ERRORE: {str(e)}", True


# ─── CONFRONTO RISPOSTE ───────────────────────────────────────────────────────
def is_correct(response_value: str, correct_value: str, question_type: str) -> bool:
    try:
        if question_type == "numeric":
            resp_num = float(response_value.strip().replace(",", "."))
            correct_num = float(correct_value.strip().replace(",", "."))
            return abs(resp_num - correct_num) <= 2.0
        elif question_type == "binary":
            r = response_value.strip().upper()
            c = correct_value.strip().upper()
            return r == c
        else:
            return response_value.strip().lower() == correct_value.strip().lower()
    except (ValueError, AttributeError):
        return False


# ─── AGGIORNAMENTO SCORE ──────────────────────────────────────────────────────
def update_scores(model_id: str, category: str, correct: bool, refused: bool):
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
        supabase.table("scores").insert({
            "model_id": model_id,
            "category": category,
            "total_questions": 1,
            "correct": 1 if correct else 0,
            "refused": 1 if refused else 0,
            "score": 100.0 if correct else 0.0
        }).execute()


# ─── VERIFICA METEO ───────────────────────────────────────────────────────────
CITY_COORDS = {
    "milano": (45.4642, 9.1900),
    "roma": (41.9028, 12.4964),
    "napoli": (40.8518, 14.2681),
    "torino": (45.0703, 7.6869),
    "firenze": (43.7696, 11.2558),
}

def verify_meteo_question(question: dict) -> str | None:
    text_lower = question["text"].lower()
    coords = None
    for city, c in CITY_COORDS.items():
        if city in text_lower:
            coords = c
            break
    if not coords:
        return None

    lat, lon = coords
    # Usa la data della deadline come data di verifica
    deadline = datetime.fromisoformat(question["deadline"].replace("Z", "+00:00"))
    date_str = deadline.strftime("%Y-%m-%d")

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max"
        f"&start_date={date_str}&end_date={date_str}"
        f"&timezone=Europe/Rome"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        temp_max = data["daily"]["temperature_2m_max"][0]
        if temp_max is None:
            return None
        return str(round(temp_max, 1))
    except Exception as e:
        print(f"  ⚠️  Verifica meteo fallita: {e}")
        return None


# ─── PIPELINE PRINCIPALE ──────────────────────────────────────────────────────
def run():
    print(f"\n🔮 SYBIL — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    models = supabase.table("ai_models").select("*").eq("active", True).execute().data
    print(f"📊 Modelli attivi: {len(models)}")

    now = datetime.now(timezone.utc).isoformat()

    # ── 1. Interroga modelli su domande aperte non scadute ──
    questions_open = (
        supabase.table("questions")
        .select("*")
        .eq("verified", False)
        .gt("deadline", now)
        .execute().data
    )
    print(f"❓ Domande aperte: {len(questions_open)}")

    for q in questions_open:
        print(f"\n📌 [{q['category'].upper()}] {q['text'][:60]}...")
        prompt = build_prompt(q["text"], q["type"])

        for model in models:
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

            supabase.table("responses").insert({
                "question_id": q["id"],
                "model_id": model["id"],
                "response_value": response_value,
                "refused": refused,
                "prompt_used": prompt
            }).execute()

    # ── 2. Verifica esiti scaduti ──
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

        if q["auto_verify"] and q["category"] == "meteo":
            correct_value = verify_meteo_question(q)

        if correct_value:
            supabase.table("questions").update({
                "correct_value": correct_value,
                "verified": True,
                "verified_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", q["id"]).execute()
            print(f"  ✅ [{q['category']}] {q['text'][:40]} → Esito: {correct_value}")
        else:
            print(f"  ⏳ [{q['category']}] Verifica manuale: {q['text'][:50]}...")

    # ── 3. Aggiorna score per TUTTE le domande verificate ──
    print("\n📈 Aggiornamento score...")
    verified_questions = (
        supabase.table("questions")
        .select("*")
        .eq("verified", True)
        .execute().data
    )

    for q in verified_questions:
        if not q["correct_value"]:
            continue

        responses = (
            supabase.table("responses")
            .select("*")
            .eq("question_id", q["id"])
            .execute().data
        )

        for resp in responses:
            # Controlla se lo score per questa risposta è già stato conteggiato
            # usando un flag nella tabella responses (campo score_counted)
            # Per ora ricalcoliamo tutto da zero ogni volta
            pass

    # Ricalcola score da zero per ogni modello/categoria
    for model in models:
        categories = set()
        all_responses = (
            supabase.table("responses")
            .select("*, questions(category, correct_value, verified, type)")
            .eq("model_id", model["id"])
            .execute().data
        )

        # Raggruppa per categoria
        cat_data = {}
        for resp in all_responses:
            q = resp.get("questions")
            if not q:
                continue
            cat = q["category"]
            if cat not in cat_data:
                cat_data[cat] = {"total": 0, "correct": 0, "refused": 0}
            cat_data[cat]["total"] += 1
            if resp["refused"]:
                cat_data[cat]["refused"] += 1
            elif q["verified"] and q["correct_value"]:
                if is_correct(resp["response_value"], q["correct_value"], q["type"]):
                    cat_data[cat]["correct"] += 1

        for cat, data in cat_data.items():
            score = round((data["correct"] / data["total"]) * 100, 2) if data["total"] > 0 else 0
            existing = (
                supabase.table("scores")
                .select("id")
                .eq("model_id", model["id"])
                .eq("category", cat)
                .execute().data
            )
            if existing:
                supabase.table("scores").update({
                    "total_questions": data["total"],
                    "correct": data["correct"],
                    "refused": data["refused"],
                    "score": score,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }).eq("model_id", model["id"]).eq("category", cat).execute()
            else:
                supabase.table("scores").insert({
                    "model_id": model["id"],
                    "category": cat,
                    "total_questions": data["total"],
                    "correct": data["correct"],
                    "refused": data["refused"],
                    "score": score
                }).execute()

        print(f"  ✓ {model['name']}: score aggiornato per {len(cat_data)} categorie")

    print("\n✅ Pipeline completata.")


if __name__ == "__main__":
    run()
