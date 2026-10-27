"""Writing short messages for the user, without any online service.

Method: pull out who the message is for (pattern matching), take the topic that the logic rules in nlu.py
derived from the user's words, then fill a ready-made template in English, Hinglish or Hindi."""
import re

# name -> (trigger phrases, {language: message})
TEMPLATES = {
    "late": (
        ["late", "running late", "be late", "a bit late", "delay", "delayed", "come late", "der ho gayi", "late ho gaya", "der se aaunga"],
        {"English": "I'm running a bit late. I'll be there as soon as I can.",
         "Hinglish": "Main thoda late ho gaya hoon. Jitni jaldi ho sakega pahunch jaunga.",
         "Hindi": "मुझे थोड़ी देर हो रही है। जैसे ही हो सकेगा मैं पहुँच जाऊँगा/जाऊँगी।"}),
    "call_back": (
        ["call back", "call you back", "call you later", "will call", "busy call later", "baad mein call", "call karunga"],
        {"English": "I can't talk right now. I'll call you back soon.",
         "Hinglish": "Abhi baat nahi kar sakta. Main thodi der mein call karta hoon.",
         "Hindi": "अभी बात नहीं हो पाएगी। मैं थोड़ी देर में कॉल करूँगा/करूँगी।"}),
    "thanks": (
        ["thank", "thanks", "thank you", "appreciate", "shukriya", "dhanyavad", "grateful"],
        {"English": "Thank you so much! I really appreciate it.",
         "Hinglish": "Bahut bahut shukriya! Main iski kadar karta hoon.",
         "Hindi": "बहुत-बहुत धन्यवाद! मैं इसकी सच में कद्र करता/करती हूँ।"}),
    "good_morning": (
        ["good morning", "morning", "suprabhat", "subah"],
        {"English": "Good morning! Have a great day.",
         "Hinglish": "Good morning! Aapka din bahut accha jaye.",
         "Hindi": "सुप्रभात! आपका दिन शुभ हो।"}),
    "good_night": (
        ["good night", "night", "sleep well", "shubh ratri", "raat"],
        {"English": "Good night! Sleep well.",
         "Hinglish": "Good night! Achhi neend aaye.",
         "Hindi": "शुभ रात्रि! अच्छी नींद आए।"}),
    "birthday": (
        ["birthday", "happy birthday", "bday", "janamdin", "birthday wish"],
        {"English": "Happy birthday! Wishing you a wonderful year ahead.",
         "Hinglish": "Janamdin mubarak ho! Aapka aane wala saal bahut accha ho.",
         "Hindi": "जन्मदिन की हार्दिक शुभकामनाएँ! आने वाला साल आपके लिए बहुत अच्छा हो।"}),
    "congrats": (
        ["congratulations", "congrats", "well done", "badhai", "promotion", "passed"],
        {"English": "Congratulations! That is wonderful news.",
         "Hinglish": "Badhai ho! Yeh bahut achhi khabar hai.",
         "Hindi": "बधाई हो! यह बहुत अच्छी खबर है।"}),
    "sorry": (
        ["sorry", "apologise", "apologize", "forgive", "maafi", "maaf"],
        {"English": "I'm sorry about that. Please forgive me.",
         "Hinglish": "Mujhe maaf kar do, mujhse galti ho gayi.",
         "Hindi": "मुझे माफ़ कर दीजिए, मुझसे गलती हो गई।"}),
    "on_my_way": (
        ["on my way", "coming", "reaching", "nikal gaya", "omw", "leaving now"],
        {"English": "I'm on my way. See you soon!",
         "Hinglish": "Main nikal chuka hoon. Jaldi milte hain!",
         "Hindi": "मैं निकल चुका/चुकी हूँ। जल्दी मिलते हैं!"}),
    "where": (
        ["where are you", "where", "kahan ho", "location"],
        {"English": "Where are you? Please let me know.",
         "Hinglish": "Aap kahan ho? Mujhe bata dijiye.",
         "Hindi": "आप कहाँ हैं? कृपया मुझे बताइए।"}),
    "ok": (
        ["okay", "ok", "fine", "agreed", "theek hai", "yes", "sure"],
        {"English": "Okay, sounds good.",
         "Hinglish": "Theek hai, chalega.",
         "Hindi": "ठीक है, चलेगा।"}),
    "cant_come": (
        ["can't come", "cannot come", "won't come", "not able to come", "won't make it", "nahi aa paunga", "not coming"],
        {"English": "I'm sorry, I won't be able to make it.",
         "Hinglish": "Maaf kijiye, main nahi aa paunga.",
         "Hindi": "क्षमा करें, मैं नहीं आ पाऊँगा/पाऊँगी।"}),
    "how_are_you": (
        ["how are you", "kaise ho", "whats up", "hows it going"],
        {"English": "How are you? It has been a while.",
         "Hinglish": "Kaise ho? Kaafi din ho gaye.",
         "Hindi": "आप कैसे हैं? काफ़ी दिन हो गए।"}),
    "miss_you": (
        ["miss you", "miss", "yaad"],
        {"English": "I miss you. Let's meet soon.",
         "Hinglish": "Mujhe aapki yaad aa rahi hai. Jaldi milte hain.",
         "Hindi": "मुझे आपकी याद आ रही है। जल्दी मिलते हैं।"}),
    "meeting": (
        ["meet", "meeting", "meet tomorrow", "milte hain", "catch up"],
        {"English": "Can we meet tomorrow? Let me know what time suits you.",
         "Hinglish": "Kya hum kal mil sakte hain? Aapko kaunsa time theek rahega?",
         "Hindi": "क्या हम कल मिल सकते हैं? आपके लिए कौन सा समय ठीक रहेगा?"}),
}

GREETING = {"English": "Hi{name},", "Hinglish": "Hi{name},", "Hindi": "नमस्ते{name},"}
SKIP_NAMES = {"that", "him", "her", "them", "me", "my", "a", "an", "the", "back", "saying", "about", "this",
              "it", "us", "you", "to", "for", "if", "i", "and", "wish", "message", "reply", "text", "write",
              "bhejo", "bhej", "do", "kar", "karo", "likho", "bolo", "bol", "de", "dena", "sorry", "thank", "thanks",
              "send", "make", "use", "open", "start", "find", "get", "go", "see", "check", "turn", "add", "share",
              "save", "create", "connect", "join", "change", "know", "learn", "do", "how", "what", "where", "be",
              "kaise", "kaisa", "kese", "kya", "kaun", "kab", "kahan", "kyun"}

NAME_PATTERNS = [
    re.compile(r"\b([A-Za-z][A-Za-z']+)\s+ko\b", re.I),
    re.compile(r"\b(?:tell|text|message|msg|ping|inform|ask|remind|call)\s+([A-Za-z][A-Za-z']+)", re.I),
    re.compile(r"\blet\s+([A-Za-z][A-Za-z']+)\s+know\b", re.I),
    re.compile(r"\b(?:wish|write to|reply to|respond to|say sorry to|thank)\s+([A-Za-z][A-Za-z']+)", re.I),
    re.compile(r"\bto\s+([A-Za-z][A-Za-z']+)", re.I),
]

_names = list(TEMPLATES.keys())


def find_recipient(text):
    for pat in NAME_PATTERNS:
        for m in pat.finditer(text):
            word = m.group(1)
            if word.lower() not in SKIP_NAMES:
                return word[:1].upper() + word[1:]
    return ""


def make_draft(text, language="English", topic=""):
    """Return (message, recipient) or (None, '') when it cannot tell what to write.
    `topic` is the template name the logic rules derived (see nlu.py); words in quotes are used as written."""
    lang = language if language in GREETING else "English"
    name = find_recipient(text)
    greet = GREETING[lang].format(name=f" {name}" if name else "")

    quoted = re.search(r"[\"“'](.{3,300}?)[\"”']\s*$|[\"“](.{3,300}?)[\"”]", text)
    if quoted:
        body = (quoted.group(1) or quoted.group(2)).strip()
        return f"{greet} {body}", name
    if topic not in TEMPLATES:
        return None, name
    return f"{greet} {TEMPLATES[topic][1][lang]}", name


def available_topics():
    return [n.replace("_", " ") for n in _names]
