import json
import re
import secrets
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from django.contrib import messages
from django.conf import settings
from django.contrib.auth import (
    login as auth_login,
    logout as auth_logout,
    update_session_auth_hash,
)
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.shortcuts import redirect, render
from django.urls import reverse
from django.http import HttpResponse

import logging
logger = logging.getLogger(__name__)

from plant_health.services import diagnose_uploaded_image, format_prediction_for_chat

from .models import ChatQuery, LANGUAGE_CHOICES, UserProfile

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

PASSWORD_PATTERN = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$")
ACCOUNT_PANELS = {"profile", "language", "themes", "password", "support"}
GOOGLE_AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"


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


def _ensure_profile(user) -> UserProfile:
    profile, _ = UserProfile.objects.get_or_create(user=user)
    return profile


def _google_auth_enabled() -> bool:
    return bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET)


def _build_unique_username(email: str, fallback_name: str = "farmer") -> str:
    base_username = email.split("@", 1)[0] if email else fallback_name
    base_username = re.sub(r"[^A-Za-z0-9_]+", "", base_username).lower() or fallback_name

    username = base_username
    suffix = 1
    while User.objects.filter(username=username).exists():
        username = f"{base_username}{suffix}"
        suffix += 1
    return username


def _build_google_redirect_uri(request) -> str:
    return settings.GOOGLE_REDIRECT_URI or request.build_absolute_uri(reverse("google_callback"))


def _build_google_authorize_url(request) -> str:
    state = secrets.token_urlsafe(24)
    request.session["google_oauth_state"] = state
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": _build_google_redirect_uri(request),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTHORIZE_ENDPOINT}?{urlencode(params)}"


def _exchange_google_code(code: str, redirect_uri: str) -> dict:
    payload = urlencode(
        {
            "code": code,
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
    ).encode("utf-8")
    request = Request(
        GOOGLE_TOKEN_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def _fetch_google_profile(access_token: str) -> dict:
    request = Request(
        GOOGLE_USERINFO_ENDPOINT,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def _upsert_google_user(profile_data: dict) -> User:
    email = (profile_data.get("email") or "").strip().lower()
    full_name = (profile_data.get("name") or profile_data.get("given_name") or "").strip()

    user = User.objects.filter(email__iexact=email).first()
    if not user:
        user = User.objects.create_user(
            username=_build_unique_username(email=email, fallback_name="googleuser"),
            email=email,
            first_name=full_name,
        )
        user.set_unusable_password()
        user.save(update_fields=["password"])
    elif full_name and not user.first_name:
        user.first_name = full_name
        user.save(update_fields=["first_name"])

    _ensure_profile(user)
    return user


def _validate_strong_password(password: str, user: User | None = None) -> list[str]:
    errors: list[str] = []

    try:
        validate_password(password, user=user)
    except ValidationError as exc:
        errors.extend(exc.messages)

    checks = [
        (len(password) >= 8, "Password must be at least 8 characters long."),
        (re.search(r"[a-z]", password), "Password must include at least one lowercase letter."),
        (re.search(r"[A-Z]", password), "Password must include at least one uppercase letter."),
        (re.search(r"\d", password), "Password must include at least one number."),
        (re.search(r"[^A-Za-z0-9]", password), "Password must include at least one symbol."),
    ]

    for passed, message in checks:
        if not passed:
            errors.append(message)

    return list(dict.fromkeys(errors))


def _render_login(request, identifier: str = ""):
    return render(
        request,
        "registration/login.html",
        {
            "identifier": identifier,
            "google_auth_enabled": _google_auth_enabled(),
        },
    )


def _render_signup(request, form_data: dict | None = None):
    return render(
        request,
        "registration/signup.html",
        {
            "form_data": form_data or {},
            "password_hint": "Use at least 8 characters with uppercase, lowercase, number, and symbol.",
            "google_auth_enabled": _google_auth_enabled(),
        },
    )


def home(request):
    if not request.session.session_key:
        request.session.create()

    session_key = request.session.session_key
    profile = _ensure_profile(request.user) if request.user.is_authenticated else None

    if request.method == "POST":
        prompt = request.POST.get("prompt", "").strip()
        uploaded_image = request.FILES.get("image")
        
        logger.info(f"POST request: prompt={bool(prompt)}, image={bool(uploaded_image)}")

        if uploaded_image:
            logger.info(f"Processing image: {uploaded_image.name} (size: {uploaded_image.size} bytes)")
            try:
                diagnosis = diagnose_uploaded_image(uploaded_image)
                logger.info(f"Diagnosis result: {diagnosis.get('status')}")
                response = format_prediction_for_chat(diagnosis)
                if not prompt:
                    prompt = f"Uploaded image: {uploaded_image.name}"
                ChatQuery.objects.create(session_key=session_key, prompt=prompt, response=response)
                logger.info(f"ChatQuery created successfully")
            except Exception as exc:
                logger.error(f"Error processing image: {exc}", exc_info=True)
                error_message = f"Error processing image: {str(exc)}"
                ChatQuery.objects.create(session_key=session_key, prompt=f"Uploaded image: {uploaded_image.name}", response=error_message)
                messages.error(request, error_message)
        elif prompt:
            response = _build_response(prompt)
            ChatQuery.objects.create(session_key=session_key, prompt=prompt, response=response)
        else:
            return redirect("home")
        return redirect("home")

    history = ChatQuery.objects.filter(session_key=session_key)
    conversation = history.order_by("created_at")

    return render(
        request,
        "core/index.html",
        {
            "history": history,
            "conversation": conversation,
            "profile": profile,
        },
    )


def login_view(request):
    """Custom login view: accepts email or phone as identifier."""
    if request.method == "POST":
        identifier = request.POST.get("identifier", "").strip()
        password = request.POST.get("password", "")

        user = None
        if identifier:
            try:
                user = User.objects.get(email__iexact=identifier)
            except User.DoesNotExist:
                profile = UserProfile.objects.filter(phone_number=identifier).first()
                if profile:
                    user = profile.user

        if user and user.check_password(password):
            _ensure_profile(user)
            auth_login(request, user)
            return redirect("home")

        messages.error(
            request,
            "Invalid credentials. Use registered email or phone and the correct password.",
        )
        return _render_login(request, identifier=identifier)

    return _render_login(request)


def google_login_view(request):
    if not _google_auth_enabled():
        messages.error(request, "Google login is not configured yet.")
        return redirect("login")

    return redirect(_build_google_authorize_url(request))


def google_callback_view(request):
    if not _google_auth_enabled():
        messages.error(request, "Google login is not configured yet.")
        return redirect("login")

    error = request.GET.get("error")
    if error:
        messages.error(request, "Google sign-in was cancelled or could not be completed.")
        return redirect("login")

    state = request.GET.get("state", "")
    expected_state = request.session.pop("google_oauth_state", "")
    if not state or state != expected_state:
        messages.error(request, "Google sign-in session expired. Please try again.")
        return redirect("login")

    code = request.GET.get("code", "")
    if not code:
        messages.error(request, "Google sign-in did not return an authorization code.")
        return redirect("login")

    try:
        token_data = _exchange_google_code(code, _build_google_redirect_uri(request))
        access_token = token_data["access_token"]
        profile_data = _fetch_google_profile(access_token)
    except (HTTPError, URLError, TimeoutError, KeyError, json.JSONDecodeError):
        messages.error(request, "Google sign-in could not be completed right now. Please try again.")
        return redirect("login")

    email = (profile_data.get("email") or "").strip()
    if not email:
        messages.error(request, "Google account email was not available for sign-in.")
        return redirect("login")

    user = _upsert_google_user(profile_data)
    auth_login(request, user)
    return redirect("home")


def signup_view(request):
    """Create User + UserProfile with strong-password validation."""
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        email = request.POST.get("email", "").strip()
        phone = request.POST.get("phone", "").strip()
        current_crop = request.POST.get("current_crop", "").strip()
        password = request.POST.get("password", "")
        password2 = request.POST.get("password2", "")

        form_data = {
            "name": name,
            "email": email,
            "phone": phone,
            "current_crop": current_crop,
        }

        if not name or not email or not password:
            messages.error(request, "Name, email, and password are required.")
            return _render_signup(request, form_data=form_data)

        if password != password2:
            messages.error(request, "Passwords do not match.")
            return _render_signup(request, form_data=form_data)

        if User.objects.filter(email__iexact=email).exists():
            messages.error(request, "A user with this email already exists.")
            return _render_signup(request, form_data=form_data)

        password_errors = _validate_strong_password(password)
        if password_errors:
            for error in password_errors:
                messages.error(request, error)
            return _render_signup(request, form_data=form_data)

        user = User.objects.create_user(
            username=_build_unique_username(email=email, fallback_name="farmer"),
            email=email,
            password=password,
            first_name=name,
        )
        UserProfile.objects.create(
            user=user,
            phone_number=phone,
            current_crop=current_crop,
        )
        auth_login(request, user)
        return redirect("home")

    return _render_signup(request)


def _redirect_with_panel(request, panel: str):
    route_name = request.resolver_match.view_name
    return redirect(f"{reverse(route_name)}?panel={panel}")


def _account_panel(request, default_panel: str):
    profile = _ensure_profile(request.user)
    panel = request.GET.get("panel", default_panel)
    if panel not in ACCOUNT_PANELS:
        panel = default_panel

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "profile":
            name = request.POST.get("name", "").strip()
            email = request.POST.get("email", "").strip()
            phone = request.POST.get("phone", "").strip()
            current_crop = request.POST.get("current_crop", "").strip()
            uploaded_profile_image = request.FILES.get("profile_image")

            if not name or not email:
                messages.error(request, "Name and email are required.")
                return _redirect_with_panel(request, "profile")

            if User.objects.filter(email__iexact=email).exclude(pk=request.user.pk).exists():
                messages.error(request, "That email address is already in use.")
                return _redirect_with_panel(request, "profile")

            request.user.first_name = name
            request.user.email = email
            request.user.save(update_fields=["first_name", "email"])

            profile.phone_number = phone
            profile.current_crop = current_crop
            updated_fields = ["phone_number", "current_crop"]
            if uploaded_profile_image:
                profile.profile_image = uploaded_profile_image
                updated_fields.append("profile_image")

            profile.save(update_fields=updated_fields)
            messages.success(
                request,
                "Profile details updated."
                if not uploaded_profile_image
                else "Profile details and image updated.",
            )
            return _redirect_with_panel(request, "profile")

        if action == "language":
            language = request.POST.get("language", "en")
            valid_languages = {code for code, _ in LANGUAGE_CHOICES}
            if language in valid_languages:
                profile.language = language
                profile.save(update_fields=["language"])
                messages.success(request, "Language preference updated.")
            else:
                messages.error(request, "Please choose a valid language.")
            return _redirect_with_panel(request, "language")

        if action == "password":
            current_password = request.POST.get("current_password", "")
            new_password = request.POST.get("new_password", "")
            confirm_password = request.POST.get("confirm_password", "")

            if request.user.has_usable_password() and not request.user.check_password(current_password):
                messages.error(request, "Current password is incorrect.")
                return _redirect_with_panel(request, "password")

            if new_password != confirm_password:
                messages.error(request, "New passwords do not match.")
                return _redirect_with_panel(request, "password")

            password_errors = _validate_strong_password(new_password, user=request.user)
            if password_errors:
                for error in password_errors:
                    messages.error(request, error)
                return _redirect_with_panel(request, "password")

            request.user.set_password(new_password)
            request.user.save()
            update_session_auth_hash(request, request.user)
            messages.success(request, "Password updated successfully.")
            return _redirect_with_panel(request, "password")

    return render(
        request,
        "core/account_hub.html",
        {
            "panel": panel,
            "profile": profile,
            "language_choices": LANGUAGE_CHOICES,
            "password_hint": "Use at least 8 characters with uppercase, lowercase, number, and symbol.",
            "has_usable_password": request.user.has_usable_password(),
        },
    )


@login_required
def account_profile_view(request):
    return _account_panel(request, default_panel="profile")


@login_required
def account_settings_view(request):
    return _account_panel(request, default_panel="language")


def logout_view(request):
    auth_logout(request)
    return redirect("home")
