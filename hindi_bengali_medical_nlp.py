"""
Hindi and Bengali Medical NLP with XLM-R and Back-translation
Fine-tuning XLM-R for Hindi intent recognition
Back-translation with BioClinicalBERT for Bengali
Author: Prof. Saptarshi Ghosh, IIT Kharagpur
Date: Nov 2023
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (XLMRobertaTokenizer, XLMRobertaModel, 
                         AutoTokenizer, AutoModel, AdamW,
                         MarianMTModel, MarianTokenizer,
                         get_linear_schedule_with_warmup)
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict
import re
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class HindiBengaliMedicalDataset(Dataset):
    """Dataset for Hindi and Bengali medical text processing"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128, language='hindi'):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language = language
        
        # Medical intent labels
        self.label2id = {
            'दवा_जानकारी': 0,      # Drug information (Hindi)
            'बीमारी_विवरण': 1,     # Disease description (Hindi)
            'इलाज_विकल्प': 2,      # Treatment options (Hindi)
            'दवा_जानकारी_bn': 0,   # Drug information (Bengali)
            'রোগ_বিবরণ': 1,        # Disease description (Bengali)
            'চিকিৎসা_বিকল্প': 2,   # Treatment options (Bengali)
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.label2id[label], dtype=torch.long)
        }

class XLMRObjective(nn.Module):
    """XLM-Roberta model for multilingual classification"""
    
    def __init__(self, model_name, num_labels, dropout_rate=0.3):
        super(XLMRObjective, self).__init__()
        self.xlmr = XLMRobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.xlmr.config.hidden_size, num_labels)
        self.num_labels = num_labels
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return loss, logits

class BackTranslationAugmenter:
    """Back-translation augmenter for Bengali medical text"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Bengali to English translation model
        self.bn_en_model_name = "Helsinki-NLP/opus-mt-bn-en"
        self.bn_en_tokenizer = MarianTokenizer.from_pretrained(self.bn_en_model_name)
        self.bn_en_model = MarianMTModel.from_pretrained(self.bn_en_model_name).to(self.device)
        
        # English to Bengali translation model
        self.en_bn_model_name = "Helsinki-NLP/opus-mt-en-bn"
        self.en_bn_tokenizer = MarianTokenizer.from_pretrained(self.en_bn_model_name)
        self.en_bn_model = MarianMTModel.from_pretrained(self.en_bn_model_name).to(self.device)
        
        # BioClinicalBERT for medical domain adaptation
        self.bioclinical_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bioclinical_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(self.device)
    
    def translate_batch(self, texts, tokenizer, model):
        """Translate batch of texts"""
        translated_texts = []
        
        for text in texts:
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                translated = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
            
            # Decode output
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            translated_texts.append(translated_text)
        
        return translated_texts
    
    def bengali_to_english(self, bengali_texts):
        """Translate Bengali texts to English"""
        print("Translating Bengali to English...")
        return self.translate_batch(bengali_texts, self.bn_en_tokenizer, self.bn_en_model)
    
    def english_to_bengali(self, english_texts):
        """Translate English texts to Bengali"""
        print("Translating English to Bengali...")
        return self.translate_batch(english_texts, self.en_bn_tokenizer, self.en_bn_model)
    
    def augment_with_bioclinicalbert(self, english_texts):
        """Enhance medical translations using BioClinicalBERT"""
        enhanced_texts = []
        
        for text in english_texts:
            # Tokenize with BioClinicalBERT
            inputs = self.bioclinical_tokenizer(
                text, 
                return_tensors="pt", 
                max_length=128, 
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get medical embeddings
            with torch.no_grad():
                outputs = self.bioclinical_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
            
            # For demonstration, we'll just return the original text
            # In practice, you'd use the embeddings to generate medically-aware translations
            enhanced_texts.append(text)
        
        return enhanced_texts
    
    def back_translate_bengali(self, bengali_texts, enhance_medical=True):
        """Perform back-translation with medical domain enhancement"""
        # Step 1: Bengali to English
        english_translations = self.bengali_to_english(bengali_texts)
        
        # Step 2: Medical domain enhancement
        if enhance_medical:
            enhanced_english = self.augment_with_bioclinicalbert(english_translations)
        else:
            enhanced_english = english_translations
        
        # Step 3: English back to Bengali
        back_translated_bengali = self.english_to_bengali(enhanced_english)
        
        return back_translated_bengali, english_translations

class HindiBengaliMedicalNLP:
    """Main class for Hindi and Bengali medical NLP"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xlmr_model = None
        self.xlmr_tokenizer = None
        self.back_translator = BackTranslationAugmenter()
        
    def create_hindi_sample_data(self, num_samples=500):
        """Create sample Hindi medical query data"""
        hindi_queries = {
            'दवा_जानकारी': [
                "मेटफॉर्मिन की खुराक क्या है?",
                "एस्पिरिन के उपयोग के बारे में बताएं",
                "इंसुलिन दवा की जानकारी",
                "लिसिनोप्रिल कैसे लें?",
                "एटोरवास्टेटिन के दुष्प्रभाव"
            ],
            'बीमारी_विवरण': [
                "मधुमेह के लक्षण क्या हैं?",
                "उच्च रक्तचाप के बारे में जानकारी",
                "कैंसर के प्रकार बताएं",
                "अस्थमा के कारण क्या हैं?",
                "गठिया कितना गंभीर है?"
            ],
            'इलाज_विकल्प': [
                "मधुमेह के लिए उपचार विकल्प",
                "उच्च रक्तचाप के लिए सर्वश्रेष्ठ चिकित्सा",
                "कैंसर उपचार के तरीके",
                "अस्थमा का इलाज कैसे करें?",
                "गठिया प्रबंधन के विकल्प"
            ]
        }
        
        texts = []
        labels = []
        
        for intent, examples in hindi_queries.items():
            for i in range(num_samples // len(hindi_queries)):
                base_query = examples[i % len(examples)]
                texts.append(base_query)
                labels.append(intent)
        
        return texts, labels
    
    def create_bengali_sample_data(self, num_samples=500):
        """Create sample Bengali medical query data"""
        bengali_queries = {
            'दवा_जानकारी_bn': [
                "মেটফর্মিনের মাত্রা কি?",
                "অ্যাসপিরিনের ব্যবহার সম্পর্কে বলুন",
                "ইনসুলিন ওষুধের তথ্য",
                "লিসিনোপ্রিল কীভাবে নেবেন?",
                "অ্যাটোরভাস্টাটিনের পার্শ্বপ্রতিক্রিয়া"
            ],
            'রোগ_বিবরণ': [
                "ডায়াবেটিসের লক্ষণগুলি কি?",
                "উচ্চ রক্তচাপ সম্পর্কে তথ্য",
                "ক্যান্সারের প্রকারগুলি বলুন",
                "হাঁপানির কারণ কী?",
                "বাত কতটা গুরুতর?"
            ],
            'চিকিৎসা_বিকল্প': [
                "ডায়াবেটিসের জন্য চিকিত্সার বিকল্প",
                "উচ্চ রক্তচাপের জন্য সেরা থেরাপি",
                "ক্যান্সার চিকিত্সার পদ্ধতি",
                "হাঁপানির চিকিত্সা কীভাবে করবেন?",
                "বাত ব্যবস্থাপনার বিকল্প"
            ]
        }
        
        texts = []
        labels = []
        
        for intent, examples in bengali_queries.items():
            for i in range(num_samples // len(bengali_queries)):
                base_query = examples[i % len(examples)]
                texts.append(base_query)
                labels.append(intent)
        
        return texts, labels
    
    def train_xlmr_hindi(self, epochs=3, batch_size=16):
        """Fine-tune XLM-R for Hindi intent recognition"""
        print("Fine-tuning XLM-R for Hindi intent recognition...")
        
        # Load tokenizer and model
        self.xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.xlmr_model = XLMRObjective('xlm-roberta-base', num_labels=3)
        self.xlmr_model.to(self.device)
        
        # Create sample data
        hindi_texts, hindi_labels = self.create_hindi_sample_data(300)
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            hindi_texts, hindi_labels, test_size=0.2, random_state=42, stratify=hindi_labels
        )
        
        # Create datasets
        train_dataset = HindiBengaliMedicalDataset(
            train_texts, train_labels, self.xlmr_tokenizer, language='hindi'
        )
        test_dataset = HindiBengaliMedicalDataset(
            test_texts, test_labels, self.xlmr_tokenizer, language='hindi'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Training setup
        optimizer = AdamW(self.xlmr_model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # Training loop
        self.xlmr_model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Hindi XLM-R Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss, _ = self.xlmr_model(input_ids, attention_mask, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.xlmr_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            print(f'Hindi XLM-R Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # Evaluate
        accuracy = self.evaluate_xlmr(test_loader)
        print(f"Hindi XLM-R Accuracy: {accuracy:.4f}")
        return accuracy
    
    def evaluate_xlmr(self, test_loader):
        """Evaluate XLM-R model performance"""
        self.xlmr_model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                _, logits = self.xlmr_model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(all_labels, all_predictions)
        return accuracy
    
    def augment_bengali_data(self, original_texts, original_labels, augmentation_factor=2):
        """Augment Bengali data using back-translation"""
        print("Augmenting Bengali data with back-translation...")
        
        augmented_texts = original_texts.copy()
        augmented_labels = original_labels.copy()
        
        # Perform back-translation
        for i in range(augmentation_factor):
            back_translated, english_translations = self.back_translator.back_translate_bengali(
                original_texts
            )
            
            augmented_texts.extend(back_translated)
            augmented_labels.extend(original_labels)
            
            print(f"Augmentation round {i+1}: Added {len(back_translated)} samples")
        
        return augmented_texts, augmented_labels
    
    def train_bengali_with_augmentation(self, epochs=3, batch_size=16):
        """Train Bengali model with back-translation augmentation"""
        print("Training Bengali model with back-translation augmentation...")
        
        # Create original Bengali data
        bengali_texts, bengali_labels = self.create_bengali_sample_data(200)
        
        # Augment data
        augmented_texts, augmented_labels = self.augment_bengali_data(
            bengali_texts, bengali_labels, augmentation_factor=2
        )
        
        print(f"Original data: {len(bengali_texts)} samples")
        print(f"Augmented data: {len(augmented_texts)} samples")
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            augmented_texts, augmented_labels, test_size=0.2, random_state=42, stratify=augmented_labels
        )
        
        # Create datasets
        train_dataset = HindiBengaliMedicalDataset(
            train_texts, train_labels, self.xlmr_tokenizer, language='bengali'
        )
        test_dataset = HindiBengaliMedicalDataset(
            test_texts, test_labels, self.xlmr_tokenizer, language='bengali'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model for Bengali
        bengali_model = XLMRObjective('xlm-roberta-base', num_labels=3)
        bengali_model.to(self.device)
        
        # Training setup
        optimizer = AdamW(bengali_model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # Training loop
        bengali_model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Bengali XLM-R Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss, _ = bengali_model(input_ids, attention_mask, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bengali_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            print(f'Bengali XLM-R Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # Evaluate
        bengali_model.eval()
        all_predictions = []
        all_labels_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                _, logits = bengali_model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())
        
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(all_labels_list, all_predictions)
        print(f"Bengali XLM-R Accuracy: {accuracy:.4f}")
        return accuracy, bengali_model
    
    def predict_hindi_intent(self, text):
        """Predict intent for Hindi medical query"""
        if self.xlmr_model is None:
            return "Model not trained"
        
        encoding = self.xlmr_tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        self.xlmr_model.eval()
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            _, logits = self.xlmr_model(input_ids, attention_mask)
            prediction = torch.argmax(logits, dim=-1).item()
        
        # Map to Hindi labels
        hindi_labels = {0: 'दवा_जानकारी', 1: 'बीमारी_विवरण', 2: 'इलाज_विकल्प'}
        return hindi_labels[prediction]
    
    def demonstrate_back_translation(self):
        """Demonstrate back-translation process"""
        print("Demonstrating Back-Translation Process:")
        print("=" * 50)
        
        sample_bengali_texts = [
            "ডায়াবেটিসের জন্য মেটফর্মিনের মাত্রা কি?",
            "উচ্চ রক্তচাপের চিকিত্সার বিকল্পগুলি কী কী?"
        ]
        
        for i, text in enumerate(sample_bengali_texts, 1):
            print(f"\nSample {i}:")
            print(f"Original Bengali: {text}")
            
            # Bengali to English
            english_translation = self.back_translator.bengali_to_english([text])[0]
            print(f"English Translation: {english_translation}")
            
            # English back to Bengali
            back_translated = self.back_translator.english_to_bengali([english_translation])[0]
            print(f"Back-translated Bengali: {back_translated}")
            
            print("-" * 40)

def main():
    """Main function to demonstrate Hindi and Bengali medical NLP"""
    nlp_system = HindiBengaliMedicalNLP()
    
    # Train Hindi intent recognition
    print("=== Hindi Intent Recognition with XLM-R ===")
    hindi_accuracy = nlp_system.train_xlmr_hindi(epochs=2)
    
    # Demonstrate back-translation for Bengali
    print("\n=== Bengali Back-Translation Demonstration ===")
    nlp_system.demonstrate_back_translation()
    
    # Train Bengali with augmentation
    print("\n=== Bengali Intent Recognition with Augmentation ===")
    bengali_accuracy, bengali_model = nlp_system.train_bengali_with_augmentation(epochs=2)
    
    # Test Hindi predictions
    print("\n=== Hindi Intent Recognition Test ===")
    hindi_test_queries = [
        "मधुमेह के लिए मेटफॉर्मिन की खुराक क्या है?",
        "उच्च रक्तचाप के उपचार विकल्प क्या हैं?",
        "कैंसर के लक्षण बताएं"
    ]
    
    for query in hindi_test_queries:
        intent = nlp_system.predict_hindi_intent(query)
        print(f"Query: {query}")
        print(f"Predicted Intent: {intent}")
        print("-" * 40)
    
    # Print results summary
    print("\n=== Results Summary ===")
    print(f"Hindi Intent Recognition Accuracy: {hindi_accuracy:.4f}")
    print(f"Bengali Intent Recognition Accuracy: {bengali_accuracy:.4f}")
    
    if hindi_accuracy >= 0.70 and bengali_accuracy >= 0.70:
        print("✅ Both models achieved good performance!")
    else:
        print("❌ Models need further improvement")

if __name__ == "__main__":
    main()