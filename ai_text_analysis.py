import os

from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient, RecognizeEntitiesAction, AnalyzeSentimentAction
from dotenv import load_dotenv
load_dotenv()

endpoint = os.environ["AZURE_AI_SERVICES_URL"]
key = os.environ["AZURE_AI_SERVICES_KEY"]
text_analytics_client = TextAnalyticsClient(endpoint, AzureKeyCredential(key))


def detect_language() -> None:
    """
    Detect the language of the text in a document.
    :return: None
    """
    print("\n -- detect_language")

    doc = [
        """
        Taking a dog for a walk provides both physical exercise and 
        mental stimulation, enriching their day and strengthening the bond between owner and pet. 
        Exploring the outdoors together, the dog's tail wags with excitement, 
        enjoying every scent and sight along the way.
        """,
        """
        Pasear a un perro proporciona tanto ejercicio físico como estimulación mental, enriqueciendo su día y fortaleciendo el vínculo entre el dueño y la mascota.
        Explorando juntos el aire libre, la cola del perro se agita con emoción, disfrutando de cada olor y vista en el camino.
        """
    ]
    result = text_analytics_client.detect_language(doc)
    for idx, doc in enumerate(result):
        print(f"Document {idx + 1} has detected language: {doc.primary_language.name}")
        print(f"Confidence score: {doc.primary_language.confidence_score}")


def sentiment_analysis() -> None:
    """
    Analyze the sentiment of the text in a document.
    :return: None
    """
    print("\n -- sentiment_analysis")

    documents = [
        """I had the best day of my life. """,
        """I think I want some ice cream.""",
        """I didn't enjoy this at all. I want my money back. """,
    ]

    result = text_analytics_client.analyze_sentiment(documents)
    docs = [doc for doc in result if not doc.is_error]

    print("Let's visualize the sentiment of each of these documents")
    for idx, doc in enumerate(docs):
        print(f"Document text: {documents[idx]}")
        print(f"Overall sentiment: {doc.sentiment}")


def recognize_entities() -> None:
    """
    Recognize entities in text.
    :return: None
    """
    print("\n -- recognize_entities")

    reviews = [
        """I work for Foo Company, and we hired Bartastic for our annual founding ceremony. The food
        was amazing and we all can't say enough good words about the quality and the level of service.""",
        """Foo Company is over the moon about the service we received from Bartastic, the best sliders ever!!!!"""
    ]

    result = text_analytics_client.recognize_entities(reviews)
    result = [review for review in result if not review.is_error]

    for idx, review in enumerate(result):
        for entity in review.entities:
            print(f"Entity: '{entity.text}' Category: '{entity.category}'")


def linked_entities() -> None:
    """
    Recognize entities and map them to their Wikipedia articles.
    :return: None
    """
    print("\n -- linked_entities")
    documents = [
        """
        Microsoft was founded by Bill Gates with some friends he met at Harvard.
        Microsoft originally moved its headquarters to Bellevue, Washington in January 1979, but is now
        headquartered in Redmond.
        """
    ]

    result = text_analytics_client.recognize_linked_entities(documents)
    docs = [doc for doc in result if not doc.is_error]
    for doc in docs:
        for entity in doc.entities:
            print(f"Entity: '{entity.name}' Mentioned: '{len(entity.matches)}' time(s) Source: '{entity.data_source}'")


def recognized_pii() -> None:
    """
    Recognize personally identifiable information (PII) in text.
    :return: None
    """
    print("\n -- recognized_pii")
    documents = [
        """Parker Doe has repaid all of their loans as of 2020-04-25.
        Their SSN is 859-98-0987. To contact them, use their phone number
        555-555-5555."""
    ]

    result = text_analytics_client.recognize_pii_entities(documents)
    docs = [doc for doc in result if not doc.is_error]

    for idx, doc in enumerate(docs):
        print(f"Document text: {documents[idx]}")
        print(f"Redacted document text: {doc.redacted_text}")
        for entity in doc.entities:
            print(f"Entity: '{entity.text}' Category: '{entity.category}' got redacted")


def get_key_phrases() -> None:
    """
    Extract key phrases from text.
    :return: None
    """
    print("\n -- get_key_phrases")
    articles = [
        """
        Redmond, WA. Employees at Microsoft can be excited about the new coffee shop that will open on campus
        once workers no longer have to work remotely...
        """
    ]
    result = text_analytics_client.extract_key_phrases(articles)
    for idx, doc in enumerate(result):
        if not doc.is_error:
            print(f"Key phrases: {", ".join(doc.key_phrases)}")


def healthcare_analysis() -> None:
    """
    Analyze healthcare data.
    :return: None
    """
    print("\n -- healthcare_analysis")
    documents = [
        """
        Patient needs to take 100 mg of ibuprofen, and 3 mg of potassium. Also needs to take
        10 mg of Zocor.
        """,
        """
        Patient needs to take 50 mg of ibuprofen, and 2 mg of Coumadin.
        """
    ]

    poller = text_analytics_client.begin_analyze_healthcare_entities(documents)
    result = poller.result()

    docs = [doc for doc in result if not doc.is_error]

    for doc in docs:
        for entity in doc.entities:
            print(f"Entity: {entity.text}")
            print(f"   Normalized Text: {entity.normalized_text}")
            print(f"   Category: {entity.category}")
            print(f"   Subcategory: {entity.subcategory}")
            print(f"   Offset: {entity.offset}")
            print(f"   Confidence score: {entity.confidence_score}")
            if entity.data_sources is not None:
                print(f"   Data Sources: {[data_source.name for data_source in entity.data_sources]}")
            if entity.assertion is not None:
                print("   Assertion:")
                print(f"      Conditionality: {entity.assertion.conditionality}")
                print(f"      Certainty: {entity.assertion.certainty}")
                print(f"      Association: {entity.assertion.association}")
        for relation in doc.entity_relations:
            print(f"Relation type: {relation.relation_type} has the following roles")
            for role in relation.roles:
                print(f"   Role '{role.name}' with entity '{role.entity.text}'")


def multi_analysis() -> None:
    """
    Analyze multiple types of data.
    :return:
    """
    print("\n -- multi_analysis")
    documents = [
        """Foo Company has the best tacos I have ever had!""",
        """Bar Company is the worst place to get a burger""",
    ]

    poller = text_analytics_client.begin_analyze_actions(documents, actions=[
        RecognizeEntitiesAction(),
        AnalyzeSentimentAction()
    ])
    document_results = poller.result()
    for doc, action_results in zip(documents, document_results):
        print(f"\nDocument text: {doc}")
        for poller in action_results:
            if poller.kind == "EntityRecognition":
                print("   Results of Recognize Entities Action:")
                for entity in poller.entities:
                    print(f"      Entity: {entity.text}")
                    print(f"         Category: {entity.category}")
                    print(f"         Confidence Score: {entity.confidence_score}")

            elif poller.kind == "SentimentAnalysis":
                print("   Results of Analyze Sentiment action:")
                print(f"      Overall sentiment: {poller.sentiment}")
                print(f"      Scores: positive={poller.confidence_scores.positive}; \
                    neutral={poller.confidence_scores.neutral}; \
                    negative={poller.confidence_scores.negative} \n"
                )

            elif poller.is_error is True:
                print(
                    f"...Is an error with code '{poller.error.code}' and message '{poller.error.message}'"
                )


if __name__ == "__main__":
    # detect_language()
    # sentiment_analysis()
    # recognize_entities()
    # linked_entities()
    # recognized_pii()
    # get_key_phrases()
    # healthcare_analysis()
    multi_analysis()