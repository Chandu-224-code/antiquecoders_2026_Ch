import re

from django.shortcuts import redirect, render

from .models import ChatQuery

GREETING_MESSAGES = {
    "hi",
    "hii",
    "hiii",
    "hello",
    "hey",
    "heyy",
    "good morning",
    "good afternoon",
    "good evening",
}


def _normalize_prompt(prompt: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", prompt.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_greeting(prompt: str) -> bool:
    normalized = _normalize_prompt(prompt)
    if not normalized:
        return False

    if normalized in GREETING_MESSAGES:
        return True

    return any(normalized.startswith(f"{greeting} ") for greeting in GREETING_MESSAGES)


def _build_response(prompt: str) -> str:
    if _is_greeting(prompt):
        return (
            "Hello! I am here to help with wheat disease detection and crop guidance. "
            "You can upload a plant image or ask a question to get started."
        )

    return f"Simulated AI analysis for: '{prompt}'. (AI integration pending)"

def home(request):
    if not request.session.session_key:
        request.session.create()
    
    session_key = request.session.session_key
    
    if request.method == "POST":
        prompt = request.POST.get('prompt', '').strip()
        if prompt:
            response = _build_response(prompt)
            ChatQuery.objects.create(session_key=session_key, prompt=prompt, response=response)
        return redirect('home')

    history = ChatQuery.objects.filter(session_key=session_key)
    conversation = history.order_by('created_at')
    
    return render(request, 'core/index.html', {'history': history, 'conversation': conversation})
