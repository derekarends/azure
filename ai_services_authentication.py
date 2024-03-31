"""
USAGE:
    python ai_services_authentication.py

    Set the environment variables with your own values before running the sample:
    1) AZURE_AI_SERVICES_URL - the endpoint to your azure ai resource.
    2) AZURE_AI_SERVICES_KEY - your azure ai API key
    3) AZURE_CLIENT_ID - the client ID of your active directory application.
    4) AZURE_TENANT_ID - the tenant ID of your active directory application.
    5) AZURE_CLIENT_SECRET - the secret of your active directory application.
    6) AZURE_KEYVAULT_URL - the URL of your Key Vault.
"""


def authentication_with_api_key() -> None:
    """
    Authenticate with the Text Analytics service using an API key.
    :return: None
    """
    print("\n -- authentication_with_api_key")
    import os
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.textanalytics import TextAnalyticsClient
    from dotenv import load_dotenv
    load_dotenv()

    endpoint = os.environ["AZURE_AI_SERVICES_URL"]
    key = os.environ["AZURE_AI_SERVICES_KEY"]

    text_analytics_client = TextAnalyticsClient(endpoint, AzureKeyCredential(key))

    # English text
    doc = [
        """
        Taking a dog for a walk provides both physical exercise and 
        mental stimulation, enriching their day and strengthening the bond between owner and pet. 
        Exploring the outdoors together, the dog's tail wags with excitement, 
        enjoying every scent and sight along the way.
        """
    ]
    result = text_analytics_client.detect_language(doc)

    print(f"Language detected: {result[0].primary_language.name}")
    print(f"Confidence score: {result[0].primary_language.confidence_score}")


def authentication_with_azure_active_directory() -> None:
    """
    Authenticate with the Text Analytics service using Azure Active Directory.
    :return: None
    """
    print("\n -- authentication_with_azure_active_directory")
    import os
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.identity import DefaultAzureCredential
    from dotenv import load_dotenv
    load_dotenv()

    endpoint = os.environ["AZURE_AI_SERVICES_URL"]
    credential = DefaultAzureCredential()

    text_analytics_client = TextAnalyticsClient(endpoint, credential=credential)

    # Spanish text
    doc = [
        """
        Sacar al perro a pasear proporciona tanto ejercicio físico como estimulación mental, 
        enriqueciendo su día y fortaleciendo el vínculo entre dueño y mascota. 
        Explorando juntos el aire libre, la cola del perro ondea con emoción, 
        disfrutando de cada olor y vista en el camino.
        """
    ]
    result = text_analytics_client.detect_language(doc)

    print(f"Language detected: {result[0].primary_language.name}")
    print(f"Confidence score: {result[0].primary_language.confidence_score}")


def authentication_with_api_key_from_vault() -> None:
    print("\n -- authentication_with_api_key_from_vault")
    import os
    from azure.identity import DefaultAzureCredential
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.keyvault.secrets import SecretClient
    from dotenv import load_dotenv
    load_dotenv()

    key_vault_url = os.environ["AZURE_KEYVAULT_URL"]
    ai_services_url = os.environ["AZURE_AI_SERVICES_URL"]

    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=key_vault_url, credential=credential)

    secret_name = "TextAnalyticsKey"
    secret = client.get_secret(secret_name)
    key = secret.value

    text_analytics_client = TextAnalyticsClient(ai_services_url, AzureKeyCredential(key))

    # Japanese text
    doc = [
        """
        犬を散歩に連れて行くことは、身体的な運動と精神的な刺激の両方を提供し、日々を豊かにし、飼い主とペットの絆を強めます。一緒に外を探索すると、犬のしっぽは興奮して振り、道中のあらゆる匂いや景色を楽しんでいます。
        """
    ]
    result = text_analytics_client.detect_language(doc)

    print(f"Language detected: {result[0].primary_language.name}")
    print(f"Confidence score: {result[0].primary_language.confidence_score}")


if __name__ == '__main__':
    authentication_with_api_key()
    authentication_with_azure_active_directory()
    authentication_with_api_key_from_vault()
