"""
Data models for the document classification system.
This file defines the structure for document types and validation rules.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DocumentType:
    """Represents a document type with its classification rules and validation requirements."""
    name: str
    keywords: List[str]
    required_fields: List[str]
    optional_fields: List[str] = None  # type: ignore
    validation_rules: Dict[str, Any] = None  # type: ignore

    def __post_init__(self):
        if self.optional_fields is None:
            self.optional_fields = []
        if self.validation_rules is None:
            self.validation_rules = {}

# Define supported document types
DOCUMENT_TYPES = {
    'electricity_bill': DocumentType(
        name='Electricity Bill',
        keywords=['kwh', 'kw-h', 'kilowatt', 'meter reading', 'electricity', 'power bill', 'energy bill', 'electric bill'],
        required_fields=['due date', 'amount', 'meter reading'],
        optional_fields=['account number', 'billing period'],
        validation_rules={
            'amount_pattern': r'\$?\d+\.?\d*',
            'date_pattern': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        }
    ),
    
    'property_tax_bill': DocumentType(
        name='Property Tax Bill',
        keywords=['property tax', 'real estate tax', 'assessment', 'municipal tax', 'council tax'],
        required_fields=['property address', 'tax amount', 'due date'],
        optional_fields=['assessment value', 'tax year'],
        validation_rules={
            'amount_pattern': r'\$?\d+\.?\d*',
            'date_pattern': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        }
    ),
    
    'birth_certificate': DocumentType(
        name='Birth Certificate',
        keywords=['birth certificate', 'certificate of birth', 'born on', 'date of birth', 'place of birth'],
        required_fields=['full name', 'date of birth', 'place of birth'],
        optional_fields=['parents names', 'registration number'],
        validation_rules={
            'date_pattern': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        }
    ),
    
    'mobile_phone_bill': DocumentType(
        name='Mobile Phone Bill',
        keywords=['mobile phone', 'cell phone', 'call charges', 'data usage', 'wireless bill', 'phone bill'],
        required_fields=['phone number', 'billing period', 'total amount'],
        optional_fields=['data usage', 'call minutes', 'text messages'],
        validation_rules={
            'phone_pattern': r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'amount_pattern': r'\$?\d+\.?\d*'
        }
    ),
    
    'water_bill': DocumentType(
        name='Water Bill',
        keywords=['water bill', 'water usage', 'water service', 'gallons', 'cubic meters', 'water department'],
        required_fields=['account number', 'usage amount', 'due date'],
        optional_fields=['billing period', 'previous reading'],
        validation_rules={
            'amount_pattern': r'\$?\d+\.?\d*',
            'date_pattern': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        }
    ),
    
    'gas_bill': DocumentType(
        name='Gas Bill',
        keywords=['gas bill', 'natural gas', 'gas service', 'therms', 'cubic feet', 'gas company'],
        required_fields=['account number', 'usage amount', 'due date'],
        optional_fields=['billing period', 'meter reading'],
        validation_rules={
            'amount_pattern': r'\$?\d+\.?\d*',
            'date_pattern': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        }
    ),
    
    'pan_card': DocumentType(
        name='PAN Card',
        keywords=['pan card', 'permanent account number', 'income tax department', 'taxpayer', 'assessee', 'father name', 'signature'],
        required_fields=['pan number', 'name', 'father name', 'date of birth'],
        optional_fields=['signature', 'photograph'],
        validation_rules={
            'pan_pattern': r'[A-Z]{5}[0-9]{4}[A-Z]{1}',
            'date_pattern': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        }
    ),
    
    'aadhaar_card': DocumentType(
        name='Aadhaar Card',
        keywords=['aadhaar', 'aadhar', 'unique identification authority', 'uidai', 'dob:', 'male', 'female', 'vid:', 'mobile no.:', '8tuic'],
        required_fields=['name', 'dob'],
        optional_fields=['mobile number', 'vid', 'address'],
        validation_rules={
            'aadhaar_pattern': r'\d{4}\s?\d{4}\s?\d{4}',
            'date_pattern': r'\d{1,2}/\d{1,2}/\d{2,4}'
        }
    )
}
