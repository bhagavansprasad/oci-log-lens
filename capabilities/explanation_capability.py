import requests


class ExplanationCapability:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000/api"

    def explain(self, input_text: str):
        response = requests.post(
            f"{self.base_url}/explain",
            json={"inputText": input_text},
            headers={"Accept": "application/json"}
        )

        response.raise_for_status()
        return response.json()
