import requests


class InvoiceCapability:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000/api"

    def get_invoice_by_id(self, invoice_id: int):
        url = f"{self.base_url}/invoice/{invoice_id}"

        response = requests.get(
            url,
            headers={"Accept": "application/json"}
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return response.json()
