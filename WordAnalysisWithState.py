import torch
from transformers import RobertaTokenizer, RobertaModel
import torch.nn.functional as F
from torch import cuda
import re
import nltk
from nltk.corpus import stopwords

# Download the stopwords data
nltk.download('stopwords', quiet=True)

device = 'cpu'

# Define the model class
class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)  # Binary classification

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# Initialize the model
model = RobertaClass()
# Load the saved model state_dict
model.load_state_dict(torch.load(r'C:\Projects\AntiCyberBullying\model_and_tokenizer\pytorch_roberta_cyberbullying.bin', map_location=torch.device('cpu')))
model.to(device)

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained(r'C:\Projects\AntiCyberBullying\model_and_tokenizer\tokenizer')

# Get the set of stopwords
stop_words = set(stopwords.words('english'))

# Function to make inference with probability
def predict(text, model, tokenizer, max_len=256):
    model.eval()
    
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )
    
    ids = inputs['input_ids'].to(device, dtype=torch.long)
    mask = inputs['attention_mask'].to(device, dtype=torch.long)
    token_type_ids = inputs['token_type_ids'].to(device, dtype=torch.long)

    # Make prediction
    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)
    
    # Get the predicted class label and its probability
    probabilities = F.softmax(outputs, dim=1)
    _, predicted = torch.max(probabilities, 1)
    
    predicted_class = predicted.item()
    predicted_probability = probabilities[0][predicted_class].item()
    
    return predicted_class, predicted_probability

def categorize_document(abusive_count, total_count):
    percentage = (abusive_count / total_count) * 100
    if percentage < 2:
        return "Good"
    elif 2 <= percentage <= 5:
        return "Average"
    else:
        return "Bad"

def preprocess_text(text):
    # Convert to lowercase and split into words
    words = text.lower().split()
    # Remove stopwords
    return [word for word in words if word not in stop_words]

def analyze_document(input_path, output_path):
    # Read the input document
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Preprocess the content
    words = preprocess_text(content)
    total_words = len(words)
    
    abusive_words = []
    
    # Analyze each word
    for word in words:
        label, probability = predict(word, model, tokenizer)
        if label == 1 and probability > 0.99:  # Assuming label 1 is for abusive words
            abusive_words.append((word, probability))
    
    # Categorize the document
    document_state = categorize_document(len(abusive_words), total_words)
    
    # Write the results to the output document
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(f"Document State: {document_state}\n\n")
        file.write(f"Total words (after removing stopwords): {total_words}\n")
        file.write(f"Number of abusive words found: {len(abusive_words)}\n")
        file.write(f"Percentage of abusive words: {(len(abusive_words) / total_words) * 100:.2f}%\n\n")
        file.write("Abusive words and their probabilities:\n")
        for word, prob in abusive_words:
            file.write(f"{word}: {prob:.4f}\n")

# Example usage
input_document_path = r"C:\Projects\AntiCyberBullying\Transcript.txt"
output_document_path = r"C:\Projects\AntiCyberBullying\document.txt"
analyze_document(input_document_path, output_document_path)