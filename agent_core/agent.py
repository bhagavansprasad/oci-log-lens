from capabilities.invoice_capability import InvoiceCapability
from capabilities.explanation_capability import ExplanationCapability


class Agent:
    def __init__(self):
        self.invoice_capability = InvoiceCapability()
        self.explanation_capability = ExplanationCapability()

    def handle_get_invoice(self, invoice_id: int):
        return self.invoice_capability.get_invoice_by_id(invoice_id)

    def handle_explanation(self, input_text: str):
        return self.explanation_capability.explain(input_text)
