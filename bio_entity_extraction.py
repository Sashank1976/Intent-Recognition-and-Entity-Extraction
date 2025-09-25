"""
Biomedical Entity Extraction with BIO Tagging
Entity types: drugs, diseases, treatments
Author: Prof. Saptarshi Ghosh, IIT Kharagpur
Date: Oct 2023
"""

import re
import json
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
import spacy
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd

class BiomedicalEntityDataset(Dataset):
    """Dataset class for biomedical entity extraction with BIO tagging"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {'O': 0, 'B-DRUG': 1, 'I-DRUG': 2, 'B-DISEASE': 3, 
                        'I-DISEASE': 4, 'B-TREATMENT': 5, 'I-TREATMENT': 6}
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        offset_mapping = encoding['offset_mapping'].squeeze()
        
        # Convert labels to token-level labels
        token_labels = self.align_labels_with_tokens(labels, offset_mapping)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(token_labels, dtype=torch.long),
            'original_text': text,
            'original_labels': labels
        }
    
    def align_labels_with_tokens(self, labels, offset_mapping):
        """Align character-level labels with token-level labels"""
        token_labels = []
        char_idx = 0
        label_idx = 0
        current_label = 'O'
        
        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:  # Special tokens
                token_labels.append(self.label2id['O'])
                continue
            
            # Find which label applies to this token
            token_label = 'O'
            while label_idx < len(labels) and labels[label_idx][1] <= start:
                label_idx += 1
            
            if label_idx < len(labels) and labels[label_idx][0] <= start < labels[label_idx][1]:
                current_label = labels[label_idx][2]
                if start == labels[label_idx][0]:  # Beginning of entity
                    token_label = f'B-{current_label}'
                else:  # Inside entity
                    token_label = f'I-{current_label}'
            else:
                token_label = 'O'
            
            token_labels.append(self.label2id[token_label])
        
        return token_labels

class BiomedicalNERModel(nn.Module):
    """Neural network model for biomedical named entity recognition"""
    
    def __init__(self, model_name, num_labels, dropout_rate=0.3):
        super(BiomedicalNERModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
        
        return loss, logits

class BiomedicalEntityExtractor:
    """Main class for biomedical entity extraction with BIO tagging"""
    
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label2id = {'O': 0, 'B-DRUG': 1, 'I-DRUG': 2, 'B-DISEASE': 3, 
                        'I-DISEASE': 4, 'B-TREATMENT': 5, 'I-TREATMENT': 6}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # Medical dictionaries for rule-based fallback
        self.drug_lexicon = self.load_medical_lexicon('drugs')
        self.disease_lexicon = self.load_medical_lexicon('diseases')
        self.treatment_lexicon = self.load_medical_lexicon('treatments')
    
    def load_medical_lexicon(self, lexicon_type):
        """Load medical lexicons for rule-based matching"""
        # In practice, these would be loaded from medical databases
        if lexicon_type == 'drugs':
            return {'aspirin', 'ibuprofen', 'metformin', 'insulin', 'atorvastatin', 
                   'lisinopril', 'metoprolol', 'simvastatin', 'omeprazole', 'losartan'}
        elif lexicon_type == 'diseases':
            return {'diabetes', 'hypertension', 'cancer', 'asthma', 'arthritis',
                   'alzheimer', 'parkinson', 'stroke', 'heart disease', 'copd'}
        elif lexicon_type == 'treatments':
            return {'surgery', 'chemotherapy', 'radiotherapy', 'physiotherapy',
                   'medication', 'vaccination', 'transplant', 'dialysis'}
        return set()
    
    def create_sample_data(self):
        """Create sample training data for demonstration"""
        texts = [
            "Patient presents with diabetes and hypertension, prescribed metformin and lisinopril",
            "Cancer treatment includes chemotherapy and radiotherapy sessions",
            "Asthma managed with inhaler and regular medication",
            "Patient with arthritis needs physiotherapy and pain management",
            "Alzheimer disease treated with medication and cognitive therapy"
        ]
        
        labels = [
            [(44, 53, 'DISEASE'), (58, 70, 'DISEASE'), (85, 94, 'DRUG'), (99, 108, 'DRUG')],
            [(0, 6, 'DISEASE'), (17, 29, 'TREATMENT'), (34, 46, 'TREATMENT')],
            [(0, 6, 'DISEASE'), (24, 32, 'TREATMENT'), (37, 50, 'TREATMENT')],
            [(14, 23, 'DISEASE'), (35, 49, 'TREATMENT'), (54, 70, 'TREATMENT')],
            [(14, 24, 'DISEASE'), (33, 43, 'TREATMENT'), (48, 67, 'TREATMENT')]
        ]
        
        return texts, labels
    
    def train(self, epochs=3, batch_size=8, learning_rate=2e-5):
        """Train the biomedical NER model"""
        texts, labels = self.create_sample_data()
        
        dataset = BiomedicalEntityDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model = BiomedicalNERModel(self.model_name, len(self.label2id))
        self.model.to(self.device)
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss, _ = self.model(input_ids, attention_mask, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def predict(self, text):
        """Predict entities in text using trained model"""
        if self.model is None:
            return self.rule_based_extraction(text)
        
        self.model.eval()
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            offset_mapping = encoding['offset_mapping'].squeeze().cpu().numpy()
            
            _, logits = self.model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
        
        # Convert token predictions to entity spans
        entities = self.extract_entities_from_predictions(text, predictions, offset_mapping)
        return entities
    
    def extract_entities_from_predictions(self, text, predictions, offset_mapping):
        """Convert token predictions to entity spans"""
        entities = []
        current_entity = None
        
        for i, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == 0 and end == 0:  # Skip special tokens
                continue
            
            label = self.id2label[pred]
            
            if label.startswith('B-'):
                # Start new entity
                if current_entity is not None:
                    entities.append(current_entity)
                entity_type = label[2:]
                current_entity = (start, end, entity_type, text[start:end])
            elif label.startswith('I-'):
                # Continue current entity
                if current_entity is not None and label[2:] == current_entity[2]:
                    # Extend entity
                    current_entity = (current_entity[0], end, current_entity[2], 
                                    text[current_entity[0]:end])
                else:
                    if current_entity is not None:
                        entities.append(current_entity)
                    current_entity = None
            else:
                # Outside entity
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity is not None:
            entities.append(current_entity)
        
        return entities
    
    def rule_based_extraction(self, text):
        """Fallback rule-based entity extraction using medical lexicons"""
        entities = []
        text_lower = text.lower()
        
        # Extract drugs
        for drug in self.drug_lexicon:
            if drug in text_lower:
                start = text_lower.find(drug)
                entities.append((start, start + len(drug), 'DRUG', drug))
        
        # Extract diseases
        for disease in self.disease_lexicon:
            if disease in text_lower:
                start = text_lower.find(disease)
                entities.append((start, start + len(disease), 'DISEASE', disease))
        
        # Extract treatments
        for treatment in self.treatment_lexicon:
            if treatment in text_lower:
                start = text_lower.find(treatment)
                entities.append((start, start + len(treatment), 'TREATMENT', treatment))
        
        return entities
    
    def evaluate(self, test_texts, test_labels):
        """Evaluate model performance"""
        self.model.eval()
        all_true_labels = []
        all_pred_labels = []
        
        for text, true_entities in zip(test_texts, test_labels):
            pred_entities = self.predict(text)
            
            # Convert to BIO format for evaluation
            true_bio = self.entities_to_bio(text, true_entities)
            pred_bio = self.entities_to_bio(text, pred_entities)
            
            all_true_labels.extend(true_bio)
            all_pred_labels.extend(pred_bio)
        
        report = classification_report(all_true_labels, all_pred_labels, 
                                     labels=list(self.label2id.keys())[1:])
        return report
    
    def entities_to_bio(self, text, entities):
        """Convert entity spans to BIO tags"""
        bio_tags = ['O'] * len(text)
        
        for start, end, entity_type, _ in entities:
            bio_tags[start] = f'B-{entity_type}'
            for i in range(start + 1, end):
                if i < len(bio_tags):
                    bio_tags[i] = f'I-{entity_type}'
        
        return bio_tags

def main():
    """Main function to demonstrate biomedical entity extraction"""
    extractor = BiomedicalEntityExtractor()
    
    print("Training biomedical entity extractor...")
    extractor.train(epochs=3)
    
    # Test the extractor
    test_texts = [
        "Patient with diabetes needs insulin treatment",
        "Cancer chemotherapy starts next week",
        "Hypertension managed with lisinopril medication"
    ]
    
    print("\nEntity Extraction Results:")
    print("=" * 50)
    
    for text in test_texts:
        entities = extractor.predict(text)
        print(f"\nText: {text}")
        print("Entities found:")
        for start, end, entity_type, entity_text in entities:
            print(f"  {entity_type}: '{entity_text}' (position {start}-{end})")
    
    # Evaluate performance
    test_texts, test_labels = extractor.create_sample_data()
    evaluation_report = extractor.evaluate(test_texts, test_labels)
    print("\nEvaluation Report:")
    print("=" * 50)
    print(evaluation_report)

if __name__ == "__main__":
    main()