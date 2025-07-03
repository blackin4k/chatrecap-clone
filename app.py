# app.py
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import re
import json
import zipfile
import io
import os
from datetime import datetime
from collections import defaultdict, Counter
import emoji
import string
import requests
import pandas as pd
import base64
import tempfile

app = Flask(__name__)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "he", "him", "his", "she", "her", "it", "its", "they", "them",
    "this", "that", "is", "am", "are", "was", "were", "be", "been", "do", "does",
    "did", "a", "an", "the", "to", "and", "but", "if", "or", "because", "as",
    "what", "which", "when", "where", "why", "how", "media","omitted","null","ok","for"
])

def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    return text.strip()

def extract_emojis(s):
    return [c for c in s if c in emoji.EMOJI_DATA]

def parse_whatsapp_chat(chat_text):
    messages = []
    pattern = re.compile(r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),?\s+(\d{1,2}:\d{2}\s?[APMapm]{2})\s+-\s+(.*?):\s+(.*)')

    for line in chat_text.split('\n'):
        match = pattern.match(line)
        if match:
            date_str, time_str, sender, message = match.groups()
            try:
                timestamp = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%y %I:%M %p")
            except:
                try:
                    timestamp = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %I:%M %p")
                except:
                    continue
            messages.append({"sender": sender, "timestamp": timestamp, "message": message})
    return messages

def analyze_messages(messages):
    total_messages = len(messages)
    sender_count = defaultdict(int)
    word_counter = Counter()
    sender_word_counter = defaultdict(Counter)
    emoji_counter = defaultdict(Counter)
    response_times = defaultdict(list)

    last_msg = {}
    for msg in messages:
        sender = msg['sender']
        message = msg['message']
        timestamp = msg['timestamp']

        sender_count[sender] += 1
        emojis = extract_emojis(message)
        for e in emojis:
            emoji_counter[sender][e] += 1

        words = clean_text(message).split()
        for word in words:
            if word not in stopwords and len(word) > 1:
                word_counter[word] += 1
                sender_word_counter[sender][word] += 1

        if last_msg and last_msg['sender'] != sender:
            delay = (timestamp - last_msg['timestamp']).total_seconds()
            if delay < 60 * 60 * 12:
                response_times[sender].append(delay)

        last_msg = msg

    avg_response_time = {
        sender: int(sum(times) / len(times)) if times else 0
        for sender, times in response_times.items()
    }

    return {
        "total_messages": total_messages,
        "messages_per_sender": dict(sender_count),
        "top_5_words": word_counter.most_common(5),
        "top_5_words_by_sender": {sender: wc.most_common(5) for sender, wc in sender_word_counter.items()},
        "emojis_by_sender": {sender: ec.most_common(3) for sender, ec in emoji_counter.items()},
        "avg_response_time": avg_response_time
    }

def get_gemini_analysis(messages):
    try:
        sample_msgs = [m['message'] for m in messages[:50]]
        prompt = """
Based on the chat messages below:
1. List 5 Red Flags about the other person.
2. List 5 Green Flags.
3. How likely are they into the user? Give a percentage from 0–100.

Messages:
""" + "\n".join(sample_msgs)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print("Gemini Error:", e)
        return "(Gemini analysis unavailable)"

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    chat_file = request.files['chat_file']
    filename = chat_file.filename.lower()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        chat_file.save(tmp)
        tmp_path = tmp.name

    if filename.endswith('.zip'):
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
            if not txt_files:
                return jsonify({'error': 'No .txt file found in ZIP'}), 400
            with zip_ref.open(txt_files[0]) as f:
                chat_data = f.read().decode('utf-8')
    elif filename.endswith(".txt"):
        chat_data = chat_file.read().decode("utf-8")
    else:
        return jsonify({"error": "Unsupported file type."}), 400

    messages = parse_whatsapp_chat(chat_data)
    result = analyze_messages(messages)
    result["gemini"] = get_gemini_analysis(messages)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
