def explain_text(input_text: str):
    text = input_text.lower()

    if "ora-01403" in text:
        return {
            "summary": "No matching data was found.",
            "possibleCause": "The requested record does not exist or access is restricted.",
            "suggestedAction": "Verify the identifier and ensure you have access permissions."
        }

    return {
        "summary": "The input was analyzed successfully.",
        "possibleCause": "The reasoning engine interpreted the message.",
        "suggestedAction": "No immediate action required."
    }
