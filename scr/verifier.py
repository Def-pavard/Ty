from flask import Blueprint, request, jsonify
import requests
import os
import logging
from datetime import datetime
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json

verifier_bp = Blueprint('verifier', __name__)
logger = logging.getLogger(__name__)

# --- MongoDB Setup ---
MONGO_URI = os.getenv("MONGO_URI") or "mongodb+srv://<db_username>:<db_password>@cluster0.rjf7z9m.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()  # Test connection
    db = client['verifier_db']
    collection = db['interactions']
except Exception as e:
    logger.error(f"Échec de la connexion à MongoDB : {e}")
    raise Exception("Impossible de se connecter à MongoDB")

# --- Load GPT-OSS 20B model and tokenizer ---
model_name = "openai/gpt-oss-20b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle GPT-OSS : {e}")
    raise Exception("Échec du chargement du modèle")

def generate_gpt_oss_response(content, max_tokens=150):
    """Génère une réponse avec le modèle GPT-OSS pour vérifier la véracité du contenu.
    
    Args:
        content (str): Contenu à vérifier.
        max_tokens (int): Nombre maximum de tokens à générer.
    
    Returns:
        str: Analyse générée par le modèle.
    """
    messages = [{"role": "user", "content": content}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)

    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    return generated_text

def log_interaction(question, answer):
    doc = {
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow()
    }
    try:
        collection.insert_one(doc)
    except Exception as e:
        logger.error(f"Erreur lors de l'insertion MongoDB : {e}")

def call_gemini_api(content, api_key, url):
    """Appelle l'API Gemini pour analyser le contenu."""
    payload = {
        "contents": [{"parts": [{"text": f"Vérifiez la véracité de ce contenu et fournissez une analyse détaillée : {content}"}]}]
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{url}?key={api_key}",
            json=payload,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if 'candidates' not in data or not data['candidates']:
            raise ValueError("Réponse Gemini malformée")
        return data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
    except requests.Timeout:
        logger.error("Timeout lors de l'appel à l'API Gemini")
        return "Erreur : Timeout de l'API Gemini"
    except requests.HTTPError as e:
        logger.error(f"Erreur HTTP Gemini : {e}")
        return f"Erreur HTTP : {str(e)}"
    except requests.ConnectionError:
        logger.error("Erreur de connexion à l'API Gemini")
        return "Erreur : Connexion à l'API Gemini échouée"
    except ValueError as e:
        logger.error(f"Erreur de format dans la réponse Gemini : {e}")
        return f"Erreur : {str(e)}"
    except requests.RequestException as e:
        logger.error(f"Gemini API error: {e}")
        return f"Erreur lors de l'analyse par Gemini : {str(e)}"

def call_grok_api(content, api_key, url, model="grok-3"):
    """Appelle l'API Grok pour analyser le contenu."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"Vérifiez la véracité de ce contenu et fournissez une analyse détaillée : {content}"}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'choices' not in data or not data['choices']:
            raise ValueError("Réponse Grok malformée")
        return data.get('choices', [{}])[0].get('message', {}).get('content', '')
    except requests.Timeout:
        logger.error("Timeout lors de l'appel à l'API Grok")
        return "Erreur : Timeout de l'API Grok"
    except requests.HTTPError as e:
        logger.error(f"Erreur HTTP Grok : {e}")
        return f"Erreur HTTP : {str(e)}"
    except requests.ConnectionError:
        logger.error("Erreur de connexion à l'API Grok")
        return "Erreur : Connexion à l'API Grok échouée"
    except ValueError as e:
        logger.error(f"Erreur de format dans la réponse Grok : {e}")
        return f"Erreur : {str(e)}"
    except requests.RequestException as e:
        logger.error(f"Grok API error: {e}")
        return f"Erreur lors de l'analyse par Grok : {str(e)}"

def generate_synthesis(content, responses, api_key, url):
    """Génère une synthèse via l'API Grok."""
    # Adjust the prompt if some responses are missing
    analyses = []
    if responses.get('openai'):
        analyses.append(f"- **GPT-OSS** : {responses['openai']}")
    if responses.get('gemini'):
        analyses.append(f"- **Gemini** : {responses['gemini']}")
    if responses.get('grok'):
        analyses.append(f"- **Grok** : {responses['grok']}")

    synthesis_prompt = f"""Vous êtes un arbitre chargé de synthétiser un débat entre les modèles d'IA disponibles sur la véracité du contenu suivant : "{content}"

Analyses fournies :
{'\n'.join(analyses)}

Votre tâche :
1. Utilisez votre fonctionnalité DeepSearch pour rechercher des informations récentes sur le web et la plateforme X afin de compléter les analyses.
2. Évaluez les points d'accord et de désaccord entre les analyses.
3. Identifiez les arguments les plus solides et les faiblesses potentielles, en intégrant les résultats de DeepSearch.
4. Fournissez une synthèse concise avec un verdict final (Vrai, Faux, À vérifier, ou Sensible) et une explication claire.
5. Citez les sources externes utilisées par DeepSearch (si disponibles).

Retournez le résultat sous forme structurée :
- **Verdict** : [Vrai/Faux/À vérifier/Sensible]
- **Synthèse** : [Explication détaillée]
- **Sources** : [Liste des sources utilisées par DeepSearch, si disponibles]
"""
    payload = {
        "model": "grok-3",
        "messages": [{"role": "user", "content": synthesis_prompt}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'choices' not in data or not data['choices']:
            raise ValueError("Réponse de synthèse Grok malformée")
        synthesis_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        # Attempt to parse as structured, but fallback to text
        try:
            # Assuming the response is markdown-like, parse it simply
            lines = synthesis_text.split('\n')
            verdict = next((line.split(': ')[1] for line in lines if line.startswith('**Verdict**')), 'Inconnu')
            synthese = next((line.split(': ')[1] for line in lines if line.startswith('**Synthèse**')), '')
            sources = next((line.split(': ')[1] for line in lines if line.startswith('**Sources**')), '')
            return {
                'Verdict': verdict,
                'Synthèse': synthese,
                'Sources': sources
            }
        except:
            return synthesis_text
    except requests.Timeout:
        logger.error("Timeout lors de la synthèse Grok")
        return "Erreur : Timeout lors de la synthèse"
    except requests.HTTPError as e:
        logger.error(f"Erreur HTTP synthèse Grok : {e}")
        return f"Erreur HTTP : {str(e)}"
    except requests.ConnectionError:
        logger.error("Erreur de connexion pour la synthèse Grok")
        return "Erreur : Connexion échouée pour la synthèse"
    except ValueError as e:
        logger.error(f"Erreur de format dans la réponse de synthèse : {e}")
        return f"Erreur : {str(e)}"
    except requests.RequestException as e:
        logger.error(f"Grok synthesis error: {e}")
        return f"Erreur lors de la synthèse : {str(e)}"

def is_limited_response(response):
    """Vérifie si la réponse de GPT-OSS indique une limitation de connaissances (ex: cutoff en 2023/2024)."""
    limitation_patterns = [
        r"knowledge cutoff",
        r"trained up to",
        r"don't have information after",
        r"my last update",
        r"limited to.*202[3-4]",
        r"pas d'informations après.*202[3-4]",
        r"connaissances jusqu'en.*202[3-4]",
        r"incapable de répondre.*actualit",
    ]
    for pattern in limitation_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return True
    return False

@verifier_bp.route('/verify', methods=['POST'])
def verify_content():
    request_id = request.headers.get('X-Request-ID', 'unknown')
    try:
        data = request.get_json()
        if not data or 'content' not in data or not isinstance(data['content'], str) or not data['content'].strip():
            logger.warning(f"[Request {request_id}] Contenu invalide ou vide")
            return jsonify({'error': 'Contenu invalide ou vide'}), 400
        if len(data['content']) > 10000:
            logger.warning(f"[Request {request_id}] Contenu trop long")
            return jsonify({'error': 'Contenu trop long (max 10000 caractères)'}), 400

        content = data['content'].strip()
        responses = {}

        # Check API keys
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            logger.error(f"[Request {request_id}] GEMINI_API_KEY is not set")
            return jsonify({'error': 'Clé API Gemini manquante'}), 500

        grok_api_key = os.getenv('GROK_API_KEY')
        if not grok_api_key:
            logger.error(f"[Request {request_id}] GROK_API_KEY is not set")
            return jupytext({'error': 'Clé API Grok manquante'}), 500

        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        grok_url = os.getenv('GROK_URL', 'https://api.x.ai/v1/chat/completions')

        # GPT-OSS as primary LLM
        try:
            local_response = generate_gpt_oss_response(f"Vérifiez la véracité de ce contenu et fournissez une analyse détaillée : {content}")
            if is_limited_response(local_response):
                logger.info(f"[Request {request_id}] Limitation de GPT-OSS détectée, relégation à Gemini comme principal")
                responses['openai'] = "Limitation de connaissances détectée : GPT-OSS ne peut pas répondre à des questions post-2024."
                # Proceed with Gemini as primary for recent topics
                responses['gemini'] = call_gemini_api(content, gemini_api_key, gemini_url)
                responses['grok'] = call_grok_api(content, grok_api_key, grok_url)
                final_synthesis = "Question d'actualité récente détectée via GPT-OSS. Analyse basée sur Gemini et Grok, avec synthèse."
            else:
                responses['openai'] = local_response
                log_interaction(content, local_response)
                # Proceed with all
                responses['gemini'] = call_gemini_api(content, gemini_api_key, gemini_url)
                responses['grok'] = call_grok_api(content, grok_api_key, grok_url)
                final_synthesis = generate_synthesis(content, responses, grok_api_key, grok_url)
        except Exception as e:
            logger.error(f"[Request {request_id}] GPT-OSS 20B génération erreur: {e}")
            responses['openai'] = f"Erreur lors de l'analyse locale GPT-OSS : {str(e)}"
            # Fallback to Gemini and Grok
            responses['gemini'] = call_gemini_api(content, gemini_api_key, gemini_url)
            responses['grok'] = call_grok_api(content, grok_api_key, grok_url)
            final_synthesis = generate_synthesis(content, responses, grok_api_key, grok_url)

        logger.info(f"[Request {request_id}] Content verification with DeepSearch completed successfully")
        return jsonify({
            'success': True,
            'openai_analysis': responses.get('openai'),
            'gemini_analysis': responses.get('gemini'),
            'grok_analysis': responses.get('grok'),
            'final_synthesis': final_synthesis
        }), 200

    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"[Request {request_id}] Erreur de validation ou de format : {e}")
        return jsonify({'error': f'Erreur de validation : {str(e)}'}), 400
    except Exception as e:
        logger.error(f"[Request {request_id}] Erreur inattendue dans verify-content endpoint: {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500