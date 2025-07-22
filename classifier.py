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
        
        # Enhanced keyword templates for stronger classification
        self.keyword_templates = {
            'Electricity Bill': [
                'electricity', 'electric', 'power', 'kWh', 'kwh', 'units consumed',
                'electric bill', 'electricity bill', 'power bill', 'energy bill',
                'meter reading', 'billing period', 'due date', 'amount due',
                'electricity charges', 'power consumption', 'energy consumption',
                'utility bill', 'electric utility', 'power company', 'energy company',
                'bill number', 'account number', 'service address', 'meter number',
                'current reading', 'previous reading', 'total units', 'tariff',
                'fuel adjustment', 'demand charges', 'fixed charges', 'variable charges'
            ],
            'Property Tax Bill': [
                'property tax', 'real estate tax', 'municipal tax', 'council tax',
                'property assessment', 'assessed value', 'taxable value', 'tax year',
                'property id', 'property number', 'parcel number', 'folio number',
                'owner name', 'property address', 'tax amount', 'tax due',
                'municipal corporation', 'city corporation', 'tax authority',
                'property type', 'residential', 'commercial', 'industrial',
                'tax rate', 'assessment roll', 'valuation', 'market value'
            ],
            'Birth Certificate': [
                'birth certificate', 'certificate of birth', 'birth record',
                'date of birth', 'place of birth', 'born on', 'born at',
                'father name', 'mother name', 'parents name', 'child name',
                'registration number', 'certificate number', 'registrar',
                'vital statistics', 'civil registration', 'birth registration',
                'hospital', 'medical facility', 'attending physician',
                'time of birth', 'gender', 'weight at birth', 'nationality'
            ],
            'Mobile Phone Bill': [
                'mobile bill', 'cell phone bill', 'cellular bill', 'phone bill',
                'mobile number', 'phone number', 'subscriber', 'customer',
                'plan details', 'monthly charges', 'usage charges', 'roaming charges',
                'data usage', 'call duration', 'sms count', 'minutes used',
                'billing cycle', 'due date', 'total amount', 'outstanding amount',
                'telecom', 'wireless', 'network provider', 'service provider',
                'airtime', 'data plan', 'voice plan', 'unlimited', 'postpaid', 'prepaid'
            ],
            'Water Bill': [
                'water bill', 'water charges', 'water supply', 'water service',
                'water consumption', 'water usage', 'gallons', 'liters', 'cubic meters',
                'meter reading', 'water meter', 'consumption units', 'billing period',
                'water authority', 'water department', 'municipal water', 'utility water',
                'service address', 'account number', 'customer id', 'connection number',
                'current reading', 'previous reading', 'sewerage charges', 'drainage charges'
            ],
            'Gas Bill': [
                'gas bill', 'natural gas', 'gas charges', 'gas consumption',
                'gas usage', 'therms', 'cubic feet', 'gas meter', 'meter reading',
                'gas company', 'gas utility', 'gas service', 'gas supply',
                'heating bill', 'fuel bill', 'energy bill', 'gas account',
                'billing period', 'due date', 'amount due', 'service address',
                'customer number', 'account number', 'current reading', 'previous reading'
            ],
            'PAN Card': [
                'permanent account number', 'pan card', 'pan number', 'income tax',
                'tax identification', 'pancard', 'pan application', 'tan number',
                'father name', 'date of birth', 'signature', 'photograph',
                'income tax department', 'government of india', 'nsdl', 'utiitsl',
                'uti technologies', 'pan services', 'application form', 'form 49a',
                'aadhaar number', 'proof of identity', 'proof of address', 'pan status'
            ],
            'Aadhaar Card': [
                'aadhaar', 'aadhar', 'unique identification', 'uidai', 'uid number',
                'aadhaar number', 'enrollment number', 'government of india',
                'unique identification authority', 'biometric', 'demographic',
                'resident', 'address proof', 'identity proof', 'aadhaar card',
                'vid number', 'virtual id', 'masked aadhaar', 'aadhaar enrollment',
                'update aadhaar', 'aadhaar authentication', 'eaadhaar', 'maadhaar',
                'download aadhaar', 'aadhaar correction', 'aadhaar update'
            ]
        }
        
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
        Perform enhanced rule-based classification using comprehensive keyword templates.
        
        Args:
            text: The text to classify
            
        Returns:
            The classification label
        """
        text_lower = text.lower()
        
        # Enhanced rule-based classification using comprehensive keyword templates
        best_match = None
        max_score = 0
        
        for doc_type, keywords in self.keyword_templates.items():
            # Calculate match score based on keyword frequency and relevance
            score = 0
            keyword_matches = 0
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    keyword_matches += 1
                    # Give higher weight to more specific keywords
                    if len(keyword.split()) > 1:  # Multi-word keywords are more specific
                        score += 2
                    else:
                        score += 1
            
            # Normalize score by total keywords to prevent bias toward categories with more keywords
            if len(keywords) > 0:
                normalized_score = (score * keyword_matches) / len(keywords)
                
                if normalized_score > max_score and keyword_matches >= 1:  # At least one keyword must match
                    max_score = normalized_score
                    best_match = doc_type
        
        # Return best match if confidence threshold is met
        if best_match and max_score > 0.02:  # Minimum confidence threshold
            logging.info(f"Rule-based classification: {best_match} (score: {max_score:.3f})")
            return best_match
        
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
        
        # Check for required fields with better name detection
        for field in doc_type.required_fields:
            field_words = field.lower().split()
            field_found = False
            
            # Special handling for name detection
            if field.lower() == 'name':
                # Look for capitalized words that could be names
                name_patterns = [
                    r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
                    r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Middle Last
                ]
                field_found = any(re.search(pattern, text) for pattern in name_patterns)
                
                # Also check for specific text patterns in Aadhaar
                if not field_found and classification == 'Aadhaar Card':
                    # Look for names after specific markers
                    lines = text.split('\n')
                    for line in lines:
                        if any(marker in line.lower() for marker in ['8tuic', 'government of india']):
                            continue
                        # Look for lines with capitalized names
                        if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+', line):
                            field_found = True
                            break
            else:
                # Default field checking
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
        
        elif classification == 'PAN Card':
            if 'government of india' not in text_lower and 'income tax' not in text_lower:
                errors.append('Indian government authority not clearly indicated')
            # Validate PAN number format if present
            pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
            if not re.search(pan_pattern, text.upper()):
                errors.append('Valid PAN number format not found (should be like ABCDE1234F)')
        
        elif classification == 'Aadhaar Card':
            if 'government of india' not in text_lower and 'uidai' not in text_lower:
                errors.append('Indian government authority not clearly indicated')
            # Validate Aadhaar number format if present
            aadhaar_pattern = r'\d{4}\s?\d{4}\s?\d{4}'
            if not re.search(aadhaar_pattern, text):
                errors.append('Valid Aadhaar number format not found (should be 12 digits)')
        
        return errors
