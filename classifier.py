"""
Document classification and validation module.
Handles rule-based classification with optional AI enhancement.
"""

import re
import os
import json
import logging
from typing import List, Tuple, Dict, Any
from models import DOCUMENT_TYPES

# Optional OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False


class DocumentClassifier:
    """
    Document classifier that uses rule-based logic and optional AI enhancement.
    """
    
    def __init__(self, use_ai: bool = True):
        """
        Initialize the classifier.
        
        Args:
            use_ai: Whether to use AI-enhanced classification (requires OpenAI API key)
        """
        self.use_ai = use_ai and OPENAI_AVAILABLE
        self.openai_client = None
        
        if self.use_ai and OPENAI_AVAILABLE:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key and OpenAI:
                self.openai_client = OpenAI(api_key=api_key)
                logging.info("OpenAI integration enabled")
            else:
                self.use_ai = False
                logging.warning("OpenAI API key not found, using rule-based classification only")
        
        logging.info(f"DocumentClassifier initialized (AI enabled: {self.use_ai})")

    def classify(self, text: str) -> str:
        """
        Classify a document based on its extracted text.
        
        Args:
            text: The extracted text from the document
            
        Returns:
            The classification label
        """
        if not text or not text.strip():
            return 'Unknown'
        
        # First try rule-based classification
        rule_based_result = self._rule_based_classify(text)
        
        # If AI is available and rule-based classification is uncertain, use AI
        if self.use_ai and rule_based_result == 'Unknown':
            try:
                ai_result = self._ai_classify(text)
                if ai_result and ai_result != 'Unknown':
                    logging.info(f"AI classification used: {ai_result}")
                    return ai_result
            except Exception as e:
                logging.error(f"AI classification failed: {str(e)}")
        
        return rule_based_result

    def _rule_based_classify(self, text: str) -> str:
        """
        Perform rule-based classification using keyword matching.
        
        Args:
            text: The text to classify
            
        Returns:
            The classification label
        """
        text_lower = text.lower()
        
        # Score each document type based on keyword matches
        scores = {}
        for doc_key, doc_type in DOCUMENT_TYPES.items():
            score = 0
            for keyword in doc_type.keywords:
                if keyword.lower() in text_lower:
                    score += 1
            scores[doc_key] = score
        
        # Find the document type with the highest score
        if scores:
            best_match = max(scores.items(), key=lambda x: x[1])
            if best_match[1] > 0:  # At least one keyword matched
                return DOCUMENT_TYPES[best_match[0]].name
        
        return 'Unknown'

    def _ai_classify(self, text: str) -> str:
        """
        Use OpenAI to classify the document.
        
        Args:
            text: The text to classify
            
        Returns:
            The classification label
        """
        if not self.openai_client:
            return 'Unknown'
        
        try:
            # Prepare the classification prompt
            document_types = [doc_type.name for doc_type in DOCUMENT_TYPES.values()]
            prompt = f"""
            Analyze the following text and classify it as one of these document types:
            {', '.join(document_types)}
            
            If the text doesn't clearly match any of these types, respond with "Unknown".
            
            Text to classify:
            {text[:2000]}  # Limit text length
            
            Respond with only the document type name or "Unknown".
            """
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a document classification expert. Classify documents accurately based on their content."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            if result:
                result = result.strip()
            else:
                result = 'Unknown'
            
            # Validate that the result is one of our known document types
            if result in [doc_type.name for doc_type in DOCUMENT_TYPES.values()]:
                return result
            
            return 'Unknown'
            
        except Exception as e:
            logging.error(f"AI classification error: {str(e)}")
            return 'Unknown'

    def get_confidence_score(self, text: str, classification: str) -> float:
        """
        Calculate a confidence score for the classification.
        
        Args:
            text: The original text
            classification: The classification result
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if classification == 'Unknown':
            return 0.0
        
        # Find the document type
        doc_type = None
        for dt in DOCUMENT_TYPES.values():
            if dt.name == classification:
                doc_type = dt
                break
        
        if not doc_type:
            return 0.0
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in doc_type.keywords if keyword.lower() in text_lower)
        
        # Base confidence on keyword matches
        confidence = min(keyword_matches / len(doc_type.keywords), 1.0)
        
        # Boost confidence if required fields are found
        required_field_matches = sum(
            1 for field in doc_type.required_fields 
            if any(word in text_lower for word in field.lower().split())
        )
        
        if required_field_matches > 0:
            confidence = min(confidence + (required_field_matches * 0.1), 1.0)
        
        return round(confidence, 2)

    def validate(self, classification: str, text: str) -> List[str]:
        """
        Validate a classified document for required fields and data integrity.
        
        Args:
            classification: The document classification
            text: The extracted text
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if classification == 'Unknown':
            errors.append('Document type could not be determined')
            return errors
        
        # Find the document type
        doc_type = None
        for dt in DOCUMENT_TYPES.values():
            if dt.name == classification:
                doc_type = dt
                break
        
        if not doc_type:
            errors.append(f'Unknown document type: {classification}')
            return errors
        
        text_lower = text.lower()
        
        # Check for required fields
        for field in doc_type.required_fields:
            field_words = field.lower().split()
            field_found = any(word in text_lower for word in field_words)
            
            if not field_found:
                errors.append(f'Missing required field: {field}')
        
        # Validate specific patterns if defined
        validation_rules = doc_type.validation_rules
        
        if 'amount_pattern' in validation_rules:
            amount_pattern = validation_rules['amount_pattern']
            if not re.search(amount_pattern, text):
                errors.append('No valid monetary amount found')
        
        if 'date_pattern' in validation_rules:
            date_pattern = validation_rules['date_pattern']
            if not re.search(date_pattern, text):
                errors.append('No valid date found')
        
        if 'phone_pattern' in validation_rules:
            phone_pattern = validation_rules['phone_pattern']
            if not re.search(phone_pattern, text):
                errors.append('No valid phone number found')
        
        # Document-specific validation
        if classification == 'Electricity Bill':
            if 'kwh' not in text_lower and 'kilowatt' not in text_lower:
                errors.append('No electricity usage information found')
        
        elif classification == 'Property Tax Bill':
            if 'tax' not in text_lower:
                errors.append('No tax information found')
        
        elif classification == 'Birth Certificate':
            if 'birth' not in text_lower:
                errors.append('Birth information not clearly indicated')
        
        return errors
