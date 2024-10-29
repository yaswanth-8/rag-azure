from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.exceptions import HttpResponseError
import os

class ContentSafetyFilter:
    def __init__(self):
        
        self.content_safety_client = ContentSafetyClient(
            endpoint=os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_CONTENT_SAFETY_KEY"))
        )

    def is_safe_content(self, text: str) -> tuple[bool, str]:
        """
        Check if the content is safe using Azure Content Safety API.
        Returns a tuple of (is_safe, reason)
        """
        try:
            # Create request
            request = AnalyzeTextOptions(text=text)

            # Analyze text
            response = self.content_safety_client.analyze_text(request)

            # It analyses the text and alots the text a value (severity) under each category

            # Check categories (violence, self-harm, sexual, hate)
            categories = {
                "Violence": response["categoriesAnalysis"][3]["severity"],
                "SelfHarm": response["categoriesAnalysis"][1]["severity"],
                "Sexual": response["categoriesAnalysis"][2]["severity"],
                "Hate": response["categoriesAnalysis"][0]["severity"]
            }

            # Define threshold (2 = Low, 3 = Medium, 4 = High)
            SEVERITY_THRESHOLD = 3

            for category, severity in categories.items():
                if severity >= SEVERITY_THRESHOLD:
                    return False, f"Content filtered due to {category} content"

            return True, "Content is safe"

        except HttpResponseError as e:
            return False, f"Error analyzing content: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
