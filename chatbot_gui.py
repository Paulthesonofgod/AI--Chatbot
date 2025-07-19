import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import aiml
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import pickle

nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.sem import Expression
from nltk.inference import ResolutionProver

read_expr = Expression.fromstring

class FootballChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Football Tactics Chatbot")
        self.root.geometry("800x600")
        
        # Initialize chatbot components
        self.setup_chatbot()
        
        # Create GUI
        self.create_widgets()
        
    def setup_chatbot(self):
        """Initialize all chatbot components"""
        try:
            # 1. AIML chatbot setup
            self.aiml_file = "aiml_patterns.aiml"
            self.kernel = self.load_aiml_kernel(self.aiml_file)
            
            # 2. Load CSV Q/A pairs
            self.qa_file = "qa_pairs.csv"
            self.qa_dict, self.df = self.load_qa_pairs(self.qa_file)
            self.questions = self.df['question'].tolist()
            self.answers = self.df['answer'].tolist()
            
            # 3. Setup similarity-based matcher
            self.vectorizer, self.tfidf_matrix, self.lemmatized_questions = self.build_tfidf_matrix(self.questions)
            
            # 4. Load knowledge base
            self.kb_file = "knowledge_base.txt"
            self.facts = self.load_knowledge_base(self.kb_file)
            
            # 5. Load CNN model
            self.cnn_model_path = "cnn_model.pkl"
            self.cnn_model, self.scaler, self.class_names = self.load_cnn_model(self.cnn_model_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error initializing chatbot: {e}")
    
    def load_aiml_kernel(self, aiml_file):
        kernel = aiml.Kernel()
        kernel.learn(aiml_file)
        return kernel

    def load_qa_pairs(self, csv_file):
        df = pd.read_csv(csv_file)
        qa_dict = dict(zip(df['question'].str.lower(), df['answer']))
        return qa_dict, df

    def lemmatize_sentence(self, sentence):
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(sentence.lower())
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmas)

    def build_tfidf_matrix(self, questions):
        lemmatized_questions = [self.lemmatize_sentence(q) for q in questions]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(lemmatized_questions)
        return vectorizer, tfidf_matrix, lemmatized_questions

    def find_best_match(self, user_input, questions, vectorizer, tfidf_matrix, threshold=0.7):
        user_lem = self.lemmatize_sentence(user_input)
        user_vec = vectorizer.transform([user_lem])
        similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        if best_score >= threshold:
            return best_idx, best_score
        return None, None

    def load_knowledge_base(self, kb_file):
        if not os.path.exists(kb_file):
            return []
        with open(kb_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return lines

    def save_knowledge_base(self, kb_file, facts):
        with open(kb_file, 'w') as f:
            for fact in facts:
                f.write(fact + '\n')

    def add_fact_to_kb(self, facts, new_fact):
        if new_fact not in facts:
            facts.append(new_fact)
            return True
        return False

    def check_fact_in_kb(self, facts, query):
        try:
            assumptions = [read_expr(fact) for fact in facts]
            goal = read_expr(query)
            proved = ResolutionProver().prove(goal, assumptions)
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

    def parse_fact_input(self, user_input):
        tokens = user_input.strip().split()
        if len(tokens) >= 6 and tokens[0].lower() == 'i' and tokens[1].lower() == 'know' and tokens[2].lower() == 'that':
            entity = tokens[3].capitalize()
            if tokens[4].lower() == 'is':
                role = tokens[5].capitalize()
                return f'{role}({entity})'
        return None

    def parse_check_input(self, user_input):
        tokens = user_input.strip().split()
        if len(tokens) >= 6 and tokens[0].lower() == 'check' and tokens[1].lower() == 'that':
            entity = tokens[2].capitalize()
            if tokens[3].lower() == 'is':
                role = tokens[4].capitalize()
                return f'{role}({entity})'
        return None

    def load_cnn_model(self, model_path):
        if not os.path.exists(model_path):
            return None, None, None
        try:
            with open(model_path, 'rb') as f:
                model, scaler, class_names = pickle.load(f)
            return model, scaler, class_names
        except Exception as e:
            return None, None, None

    def classify_image(self, model, scaler, class_names, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((32, 32))
            img_gray = img.convert('L')
            img_array = np.array(img_gray).flatten() / 255.0
            img_array = img_array.reshape(1, -1)
            img_scaled = scaler.transform(img_array)
            prediction = model.predict(img_scaled)[0]
            confidence = model.predict_proba(img_scaled).max()
            return class_names[prediction], confidence
        except Exception as e:
            return f"Error classifying image: {e}", 0.0

    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(main_frame, width=70, height=25, wrap=tk.WORD)
        self.chat_display.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Input field
        self.input_field = ttk.Entry(main_frame, width=50)
        self.input_field.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.input_field.bind('<Return>', self.send_message)
        
        # Send button
        send_button = ttk.Button(main_frame, text="Send", command=self.send_message)
        send_button.grid(row=1, column=1, padx=(0, 10))
        
        # Classify image button
        classify_button = ttk.Button(main_frame, text="Classify Image", command=self.classify_image_gui)
        classify_button.grid(row=1, column=2)
        
        # Welcome message
        self.add_message("Bot", "Welcome to the Football Tactics Chatbot! Ask me about football tactics, player positions, or use 'Classify Image' to analyze football images.")
        
    def add_message(self, sender, message):
        """Add a message to the chat display"""
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.see(tk.END)
        
    def send_message(self, event=None):
        """Process user input and generate response"""
        user_input = self.input_field.get().strip()
        if not user_input:
            return
            
        # Clear input field
        self.input_field.delete(0, tk.END)
        
        # Display user message
        self.add_message("You", user_input)
        
        # Process the input
        response = self.process_input(user_input)
        
        # Display bot response
        self.add_message("Bot", response)
        
    def process_input(self, user_input):
        """Process user input and return appropriate response"""
        # 1. Check direct CSV Q/A match
        answer = self.qa_dict.get(user_input.lower())
        if answer:
            return answer
            
        # 2. Similarity-based matching
        best_idx, best_score = self.find_best_match(user_input, self.questions, self.vectorizer, self.tfidf_matrix)
        if best_idx is not None:
            return f"{self.answers[best_idx]} (matched with similarity {best_score:.2f})"
            
        # 3. Logic reasoning: Add fact
        fact = self.parse_fact_input(user_input)
        if fact:
            if self.add_fact_to_kb(self.facts, fact):
                self.save_knowledge_base(self.kb_file, self.facts)
                return f"Fact '{fact}' added to knowledge base."
            else:
                return f"Fact '{fact}' already exists in knowledge base."
                
        # 4. Logic reasoning: Check fact
        check = self.parse_check_input(user_input)
        if check:
            result = self.check_fact_in_kb(self.facts, check)
            return result
            
        # 5. Check AIML patterns
        aiml_response = self.kernel.respond(user_input)
        if aiml_response:
            return aiml_response
            
        # 6. Fallback
        return "Sorry, I don't know the answer to that yet."
        
    def classify_image_gui(self):
        """Open file dialog to select image for classification"""
        if self.cnn_model is None:
            messagebox.showwarning("Warning", "CNN model not loaded. Please train the model first.")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image to Classify",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            result, confidence = self.classify_image(self.cnn_model, self.scaler, self.class_names, file_path)
            self.add_message("You", f"classify image {file_path}")
            self.add_message("Bot", f"This image is classified as: {result} (confidence: {confidence:.2f})")

def main():
    root = tk.Tk()
    app = FootballChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 