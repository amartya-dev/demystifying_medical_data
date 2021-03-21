import logging
import math
from newspaper import Article, ArticleException
from collections import Counter
from spellchecker import SpellChecker
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from textblob import TextBlob
import spacy
from readability import Readability
from readability.exceptions import ReadabilityException
from bs4 import BeautifulSoup as soup
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
import tldextract
from spacy.tokens import Doc

log = logging.getLogger(__name__)


from app.models import PIPELINE_STAGES as STAGE


class CredibilityPredictor(object):
    """
    Analyzes the link for credibility
    """

    def __init__(self, nlp):
        pass

    # Helper functions to extract features

    # Get the content features
    def get_article_content(url):
        article = Article(url)
        article.download()
        article.parse()
        return {
            "authors": article.authors,
            "content": article.text,
        }

    def get_punctuations(content):
        counts = Counter(content)
        finders_list = "!,.?"
        required_counts = {k: v for k, v in counts.items() if k in finders_list}
        # get the required punctuations
        punctuation_features = {
            "exclamations": required_counts.get("!", 0),
            "commas": required_counts.get(",", 0),
            "dots": required_counts.get(".", 0),
            "questions": required_counts.get("?", 0),
        }
        return punctuation_features

    def get_sentences(content):
        return list(
            filter(
                lambda s: s != "", list(map(lambda s: s.strip(), content.split("\n")))
            )
        )

    def get_readability(content):
        r = Readability(content)
        try:
            return r.smog().score
        except ReadabilityException:
            return None

    def get_word_related_stats(sentences):
        nlp = spacy.load("en_core_web_sm")
        content = ". ".join(sentences)
        doc = nlp(content)
        num_words = len(doc)
        c = Counter(([token.pos_ for token in doc]))
        num_verbs = c.get("VERB")
        num_nouns = c.get("NOUN")
        num_adverbs = c.get("ADP")
        num_determiners = c.get("DET")
        all_words = [
            token.lemma_ for token in doc if token.is_alpha and not token.is_stop
        ]
        spelling_check_words = [
            token.lemma_
            for token in doc
            if token.is_alpha and not token.is_stop and not token.pos_ == "PROPN"
        ]
        all_words_counter = Counter(all_words)
        entropy = 0
        n = len(all_words)
        for word_count in all_words_counter:
            fi = all_words_counter.get(word_count)
            entropy += fi * (math.log10(n) - math.log10(fi))
        return {
            "words": list(set(all_words)),
            "spell_check": list(set(spelling_check_words)),
            "num_words": num_words,
            "num_nouns": num_nouns,
            "num_verbs": num_verbs,
            "num_adverbs": num_adverbs,
            "num_determiners": num_determiners,
            "text_entropy": entropy,
        }

    def get_spelling_errors(words):
        checker = SpellChecker()
        return checker.unknown(words)

    def get_sentiments_and_subjectivity(sentences):
        dataset_name = "imdb"
        saved_model_path = "./{}_bert".format(dataset_name.replace("/", "_"))
        reloaded_model = tf.saved_model.load(saved_model_path)
        sentiments = tf.sigmoid(reloaded_model(tf.constant(sentences)))
        sentiments = list(
            map(
                lambda sentiment: round(sentiment),
                sentiments.numpy().flatten().tolist(),
            )
        )
        subjectivity = [round(TextBlob(i).sentiment.subjectivity) for i in sentences]

        sentiment_counter = Counter(sentiments)
        num_positive_sentences = sentiment_counter.get(1)
        num_negative_sentences = sentiment_counter.get(0)

        subjectivity_counter = Counter(subjectivity)
        num_subjective_sentences = sentiment_counter.get(1)
        num_objective_sentences = sentiment_counter.get(0)
        return {
            "polarity": sentiment_counter.most_common(1)[0][0],
            "num_positive": num_positive_sentences,
            "num_negative": num_negative_sentences,
            "num_subjective": num_subjective_sentences,
            "num_objective": num_objective_sentences,
        }

    def get_content_features(self, url):
        article = self.get_article_content(url)
        content = article.get("content")
        punctuations = self.get_punctuations(content)
        sentences = self.get_sentences(content)
        sentiment_dic = self.get_sentiments_and_subjectivity(sentences)
        words_metrics = self.get_word_related_stats(sentences)
        spelling_errors = self.get_spelling_errors(words_metrics.get("words"))
        smog = self.get_readability(content)

        return {
            "content": content,
            "punctuations": punctuations,
            "sentences": sentences,
            "sentiment_dic": sentiment_dic,
            "words_metrics": words_metrics,
            "spelling_errors": spelling_errors,
            "smog": smog,
        }

    # Gettint the appearance features
    def get_ads_blacklist(file_path):
        with open(file_path) as file:
            return file.read().splitlines()

    def get_ads(self, url):
        add_urls = self.get_ads_blacklist("add_urls.txt")

        # Find all the iframes that have add_urls as source
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.headless = True
        browser = webdriver.Chrome("./chromedriver", options=chrome_options)
        browser.set_page_load_timeout(30)
        try:
            browser.get(url)
        except TimeoutException:
            return {"ads": [], "num_ads": 0}
        parsed_html = soup(browser.page_source, "html.parser")

        num_ads = 0
        ads = []
        for iframe in parsed_html.findAll("iframe"):
            for url in add_urls:
                if iframe.get("src") != None and url in iframe.get("src"):
                    num_ads += 1
                    ads.append(iframe)
            if iframe.get("name") != None and "google_ads" in iframe.get("name"):
                num_ads += 1
                ads.append(iframe)
        return {"ads": ads, "num_ads": num_ads}

    # Meta info
    def get_domain_type(url):
        return tldextract.extract(url).suffix

    # Get social popularity
    def get_social_info(url):
        response = requests.get(
            f"http://count-server.sharethis.com/v2.0/get_counts?url={url}"
        )
        info = response.json()
        return info.get("shares")

    # General popularity
    def get_alexa_rank(url):
        return soup(
            requests.get(f"http://data.alexa.com/data?cli=10&dat=s&url={url}").content,
            "xml",
        ).find("REACH")["RANK"]

    # A master function to get all required features for a url
    def get_features(self, url):
        try:
            logging.info(f"Getting content features for {url}")
            content_features = self.get_content_features(url)
            logging.info(f"Getting ads for {url}")
            ads = self.get_ads(url)
            logging.info(f"Getting domain type for {url}")
            domain_type = self.get_domain_type(url)
            logging.info(f"Getting social information for {url}")
            social_information = self.get_social_info(url)
            logging.info(f"Getting alexa rank for {url}")
            alexa_rank = self.get_alexa_rank(url)
            print(f"Getting smog score for {url}")
            smog_score = self.get_readability(content_features.get("content"))

            return {
                "exclamations": content_features.get("punctuations").get(
                    "exclamations"
                ),
                "commas": content_features.get("punctuations").get("commas"),
                "dots": content_features.get("punctuations").get("dots"),
                "questions": content_features.get("punctuations").get("questions"),
                "token_count": content_features.get("words_metrics").get("num_words"),
                "polarity": content_features.get("sentiment_dic").get("polarity"),
                "positive": content_features.get("sentiment_dic").get("num_positive"),
                "negative": content_features.get("sentiment_dic").get("num_negative"),
                "subjective": content_features.get("sentiment_dic").get(
                    "num_subjective"
                ),
                "objective": content_features.get("sentiment_dic").get("num_objective"),
                "spelling_errors": len(content_features.get("spelling_errors")),
                "text_complexity": content_features.get("words_metrics").get(
                    "text_entropy"
                ),
                "smog": smog_score,
                "nouns": content_features.get("words_metrics").get("num_nouns"),
                "verbs": content_features.get("words_metrics").get("num_verbs"),
                "adverbs": content_features.get("words_metrics").get("num_adverbs"),
                "determiners": content_features.get("words_metrics").get(
                    "num_determiners"
                ),
                "ad_count": ads.get("num_ads"),
                "domain_type": domain_type,
                "fb_shares": social_information.get("facebook"),
                "fb_like": social_information.get("fb_like"),
                "linkedin_shares": social_information.get("linkedin"),
                "alexa_rank": alexa_rank,
            }
        except ArticleException:
            return None

    def _calculate_score(self, doc: Doc):
        """
        Getter method. Calculates the score.
        """

        assert doc.has_extension(STAGE.CREDIBILITY_SCORE)
        try:
            result = self.get_features()
            return result

        except Exception as e:
            raise Exception(e)
