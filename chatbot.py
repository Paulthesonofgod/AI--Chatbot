import aiml
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model
from PIL import Image
import os

nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.sem import Expression
from nltk.inference import ResolutionProver

read_expr = Expression.fromstring


def load_aiml_kernel(aiml_file):
    kernel = aiml.Kernel()
    kernel.learn(aiml_file)
    return kernel

def load_qa_pairs(csv_file):
    df = pd.read_csv(csv_file)
    qa_dict = dict(zip(df['question'].str.lower(), df['answer']))
    return qa_dict, df

def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(sentence.lower())
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmas)

def build_tfidf_matrix(questions):
    lemmatized_questions = [lemmatize_sentence(q) for q in questions]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(lemmatized_questions)
    return vectorizer, tfidf_matrix, lemmatized_questions

def find_best_match(user_input, questions, vectorizer, tfidf_matrix, threshold=0.7):
    user_lem = lemmatize_sentence(user_input)
    user_vec = vectorizer.transform([user_lem])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    if best_score >= threshold:
        return best_idx, best_score
    return None, None

# --- Logic Reasoning Functions ---
def load_knowledge_base(kb_file):
    if not os.path.exists(kb_file):
        return []
    with open(kb_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return lines

def save_knowledge_base(kb_file, facts):
    with open(kb_file, 'w') as f:
        for fact in facts:
            f.write(fact + '\n')

def add_fact_to_kb(facts, new_fact):
    if new_fact not in facts:
        facts.append(new_fact)
        return True
    return False

def check_fact_in_kb(facts, query):
    # Use NLTK's logic prover
    try:
        assumptions = [read_expr(fact) for fact in facts]
        goal = read_expr(query)
        # Try to prove the goal
        proved = ResolutionProver().prove(goal, assumptions)
        # Try to prove the negation
        negated = read_expr(f'-({query})')
        disproved = ResolutionProver().prove(negated, assumptions)
        if proved:
            return 'Correct'
        elif disproved:
            return 'Incorrect'
        else:
            return "I don't know"
    except Exception as e:
        return f"Error in logic: {e}"

def parse_fact_input(user_input):
    # "I know that Messi is a forward" -> Forward(Messi)
    tokens = user_input.strip().split()
    if len(tokens) >= 6 and tokens[0].lower() == 'i' and tokens[1].lower() == 'know' and tokens[2].lower() == 'that':
        entity = tokens[3].capitalize()
        if tokens[4].lower() == 'is':
            role = tokens[5].capitalize()
            return f'{role}({entity})'
    return None

def parse_check_input(user_input):
    # "Check that Messi is a forward" -> Forward(Messi)
    tokens = user_input.strip().split()
    if len(tokens) >= 6 and tokens[0].lower() == 'check' and tokens[1].lower() == 'that':
        entity = tokens[2].capitalize()
        if tokens[3].lower() == 'is':
            role = tokens[4].capitalize()
            return f'{role}({entity})'
    return None

def load_cnn_model(model_path):
    if not os.path.exists(model_path):
        print(f"Warning: CNN model file '{model_path}' not found. Image classification will be disabled.")
        return None
    return load_model(model_path)

def classify_image(model, image_path, class_indices):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((64, 64))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr)
        class_idx = np.argmax(preds)
        # Reverse class_indices dict to get label
        idx_to_class = {v: k for k, v in class_indices.items()}
        return idx_to_class.get(class_idx, 'Unknown')
    except Exception as e:
        return f"Error classifying image: {e}"

def main():
    # 1. AIML chatbot setup
    aiml_file = "aiml_patterns.aiml"
    kernel = load_aiml_kernel(aiml_file)

    # 2. Load CSV Q/A pairs
    qa_file = "qa_pairs.csv"
    qa_dict, df = load_qa_pairs(qa_file)
    questions = df['question'].tolist()
    answers = df['answer'].tolist()

    # 3. Setup similarity-based matcher
    vectorizer, tfidf_matrix, lemmatized_questions = build_tfidf_matrix(questions)

    # 4. Load knowledge base
    kb_file = "knowledge_base.txt"
    facts = load_knowledge_base(kb_file)

    # 5. Load CNN model
    cnn_model_path = "cnn_model.h5"
    cnn_model = load_cnn_model(cnn_model_path)
    # Set class indices (must match those printed during training)
    class_indices = {'Defender': 0, 'Forward': 1, 'Goalkeeper': 2, 'Midfielder': 3}
    # If you have the actual mapping from your training, update this dict accordingly.

    print("Football Tactics Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        # 0. Image classification command
        if user_input.lower().startswith('classify image '):
            if cnn_model is None:
                print("Bot: Image classification model not loaded.")
                continue
            image_path = user_input[len('classify image '):].strip()
            if not os.path.exists(image_path):
                print(f"Bot: Image file '{image_path}' not found.")
                continue
            result = classify_image(cnn_model, image_path, class_indices)
            print(f"Bot: This image is classified as: {result}")
            continue
        # 1. Check direct CSV Q/A match
        answer = qa_dict.get(user_input.lower())
        if answer:
            print(f"Bot: {answer}")
            continue
        # 2. Similarity-based matching
        best_idx, best_score = find_best_match(user_input, questions, vectorizer, tfidf_matrix)
        if best_idx is not None:
            print(f"Bot: {answers[best_idx]} (matched with similarity {best_score:.2f})")
            continue
        # 3. Logic reasoning: Add fact
        fact = parse_fact_input(user_input)
        if fact:
            if add_fact_to_kb(facts, fact):
                save_knowledge_base(kb_file, facts)
                print(f"Bot: Fact '{fact}' added to knowledge base.")
            else:
                print(f"Bot: Fact '{fact}' already exists in knowledge base.")
            continue
        # 4. Logic reasoning: Check fact
        check = parse_check_input(user_input)
        if check:
            result = check_fact_in_kb(facts, check)
            print(f"Bot: {result}")
            continue
        # 5. Check AIML patterns
        aiml_response = kernel.respond(user_input)
        if aiml_response:
            print(f"Bot: {aiml_response}")
            continue
        # 6. Fallback
        print("Bot: Sorry, I don't know the answer to that yet.")

if __name__ == "__main__":
    main() 