from django.test import TestCase
from django.urls import reverse

from .models import ChatQuery


class GreetingResponseTests(TestCase):
    def test_greeting_message_gets_greeting_response(self):
        response = self.client.post(reverse("home"), {"prompt": "hii"}, follow=True)

        self.assertEqual(response.status_code, 200)
        chat = ChatQuery.objects.get()
        self.assertIn("Hello!", chat.response)
        self.assertContains(response, "Hello!")

    def test_non_greeting_keeps_default_simulated_response(self):
        self.client.post(reverse("home"), {"prompt": "Check my wheat leaf"}, follow=True)

        chat = ChatQuery.objects.get()
        self.assertIn("Simulated AI analysis", chat.response)
