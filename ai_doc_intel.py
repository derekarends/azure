"""
USAGE:
    python ai_services_authentication.py

    Set the environment variables with your own values before running the sample:
    1) AZURE_AI_SERVICES_URL - the endpoint to your azure ai resource.
    2) AZURE_AI_SERVICES_KEY - your azure ai API key
"""


def document_intelligence() -> None:
    print("\n -- document_intelligence")
    import os
    import requests

    from azure.core.credentials import AzureKeyCredential
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from dotenv import load_dotenv
    load_dotenv()

    # Get the endpoint and key from the environment
    endpoint = os.environ["AZURE_AI_SERVICES_URL"]
    key = os.environ["AZURE_AI_SERVICES_KEY"]

    # Create a client
    credential = AzureKeyCredential(key)
    client = DocumentAnalysisClient(endpoint, credential)

    # Sample document
    file_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-layout.pdf"
    file_bytes = requests.get(file_url).content

    # Analyze the document
    poller = client.begin_analyze_document(model_id="prebuilt-layout", document=file_bytes)
    result = poller.result()

    # Process the result
    for page in result.pages:
        print(f"Document Page {page.page_number} has {len(page.lines)} line(s), {len(page.words)} word(s)")

        for i, line in enumerate(page.lines):
            print(f"Line {i}: ")
            print(f"Content: '{line.content}'")
            print("Bounding polygon, with points ordered clockwise: ")
            for j in range(len(line.polygon)):
                print(f" ({line.polygon[j].x}, {line.polygon[j].y})")

        for i, selection_mark in enumerate(page.selection_marks):
            print(f"Selection Mark {i} is {selection_mark.state}.")
            print(f"State: {selection_mark.state}")
            print("Bounding polygon, with points ordered clockwise: ")
            for j in range(len(selection_mark.polygon)):
                print(f" ({selection_mark.polygon[j].x}, {selection_mark.polygon[j].y})")

    for i, paragraph in enumerate(result.paragraphs):
        print(f"Paragraph {i}: ")
        print(f"Content: {paragraph.content}")

    for style in result.styles:
        if style.is_handwritten and style.confidence > 0.8:
            print("Handwritten content found: ")
            for span in style.spans:
                print(f"{result.content[span.offset:span.offset+span.length]}")

    for i, table in enumerate(result.tables):
        print(f"Table {i} has {table.row_count} rows and {table.column_count} columns.")
        for cell in table.cells:
            print(f" Cell ({cell.row_index}, {cell.column_index}) is a '{cell.kind}' with content: {cell.content}")


if __name__ == '__main__':
    document_intelligence()
