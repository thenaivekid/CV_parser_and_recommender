"""
Resume Parser using LangChain with Azure OpenAI or Google Gemini
Reads PDF resumes and converts them to structured JSON format with robust error handling
"""

import os
import json
import logging
import re
from typing import Literal, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class Location(BaseModel):
    """Location information"""
    city: str = Field(default="", description="City name")
    countryCode: str = Field(default="", description="Two-letter country code")
    
    @field_validator('countryCode')
    @classmethod
    def validate_country_code(cls, v):
        """Ensure country code is uppercase and 2 letters"""
        if v and len(v) == 2:
            return v.upper()
        return v


class Basics(BaseModel):
    """Basic information about the person"""
    name: str = Field(default="", description="Full name")
    email: str = Field(default="", description="Email address")
    phone: str = Field(default="", description="Phone number")
    location: Location = Field(default_factory=Location, description="Location information")
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """Basic email validation"""
        if v and '@' in v:
            return v.strip()
        return v


class WorkExperience(BaseModel):
    """Work experience entry"""
    company: str = Field(description="Company name")
    position: str = Field(description="Job position/title")
    startDate: str = Field(description="Start date in YYYY-MM format")
    endDate: str = Field(description="End date in YYYY-MM format or 'Present'")
    summary: str = Field(description="Summary of responsibilities and achievements")


class Education(BaseModel):
    """Education entry"""
    institution: str = Field(description="Institution name")
    area: str = Field(description="Field of study")
    studyType: str = Field(description="Degree type (e.g., Bachelor of Science, Master of Science)")
    startDate: str = Field(description="Start date in YYYY-MM-DD format")
    endDate: str = Field(description="End date in YYYY-MM-DD format")


class Skills(BaseModel):
    """Skills categorized as technical and soft skills"""
    technical: list[str] = Field(default=[], description="List of technical skills")
    soft: list[str] = Field(default=[], description="List of soft skills")


class Certificate(BaseModel):
    """Certificate entry"""
    name: str = Field(description="Certificate name")
    issuer: str = Field(description="Issuing organization")
    date: str = Field(description="Issue date in YYYY-MM-DD format")


class Achievement(BaseModel):
    """Achievement or award entry"""
    title: str = Field(description="Achievement/award title")
    awarder: str = Field(description="Organization that gave the award")
    date: str = Field(description="Award date in YYYY-MM-DD format")
    description: str = Field(description="Achievement description")


class Language(BaseModel):
    """Language proficiency entry"""
    language: str = Field(description="Language name")
    fluency: str = Field(description="Fluency level (e.g., Native, Fluent, Professional Working Proficiency, Limited Working Proficiency)")


class ResumeData(BaseModel):
    """Complete resume data structure"""
    basics: Basics = Field(description="Basic personal information")
    summary: str = Field(default="", description="Professional summary or objective statement")
    work: list[WorkExperience] = Field(default=[], description="Work experience")
    education: list[Education] = Field(default=[], description="Education history")
    skills: Skills = Field(description="Technical and soft skills")
    certifications: list[Certificate] = Field(default=[], description="Certifications")
    achievements: list[Achievement] = Field(default=[], description="Achievements and awards")
    languages: list[Language] = Field(default=[], description="Languages spoken")


class ResumeParser:
    """Parser for converting PDF resumes to structured JSON with robust error handling"""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.pdf'}
    
    # Maximum file size (20MB)
    MAX_FILE_SIZE = 20 * 1024 * 1024
    
    def __init__(self, provider: Literal["azure", "gemini"] = "azure"):
        """
        Initialize the resume parser
        
        Args:
            provider: LLM provider to use ("azure" for Azure OpenAI or "gemini" for Google Gemini)
            
        Raises:
            ValueError: If provider is not supported
            EnvironmentError: If required environment variables are missing
        """
        self.provider = provider
        self._validate_environment()
        self.llm = self._initialize_llm()
        self.parser = JsonOutputParser(pydantic_object=ResumeData)
        logger.info(f"Initialized ResumeParser with provider: {provider}")
    
    def _validate_environment(self):
        """Validate that required environment variables are set"""
        if self.provider == "azure":
            required_vars = [
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_VERSION",
                "AZURE_OPENAI_DEPLOYMENT",
                "AZURE_OPENAI_API_KEY"
            ]
        elif self.provider == "gemini":
            required_vars = ["GOOGLE_API_KEY"]
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables for {self.provider}: {', '.join(missing_vars)}"
            )
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on provider"""
        try:
            if self.provider == "azure":
                # Note: o3-mini doesn't support temperature parameter
                return AzureChatOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY")
                )
            elif self.provider == "gemini":
                return ChatGoogleGenerativeAI(
                    model="gemini-3-pro-preview",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0
                )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _validate_pdf_file(self, pdf_path: str) -> None:
        """Validate PDF file before processing
        
        Args:
            pdf_path: Path to the PDF file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a PDF or exceeds size limit
        """
        path = Path(pdf_path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}. Only PDF files are supported.")
        
        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"File size ({file_size / 1024 / 1024:.2f}MB) exceeds maximum allowed size "
                f"({self.MAX_FILE_SIZE / 1024 / 1024:.2f}MB)"
            )
        
        if file_size == 0:
            raise ValueError("PDF file is empty")
    
    def read_pdf(self, pdf_path: str) -> str:
        """Read text content from a PDF file with robust error handling
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid or corrupted
            PdfReadError: If PDF cannot be read
        """
        try:
            # Validate file first
            self._validate_pdf_file(pdf_path)
            logger.info(f"Reading PDF: {pdf_path}")
            
            # Read PDF with error handling
            reader = PdfReader(pdf_path)
            
            # Check if PDF has pages
            if len(reader.pages) == 0:
                raise ValueError("PDF has no pages")
            
            text = ""
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i+1}: {e}")
                    continue
            
            # Clean extracted text
            text = self._clean_text(text)
            
            if not text.strip():
                raise ValueError("No text could be extracted from PDF. The file may be image-based or corrupted.")
            
            logger.info(f"Successfully extracted {len(text)} characters from {len(reader.pages)} pages")
            return text
            
        except PdfReadError as e:
            logger.error(f"PDF read error: {e}")
            raise ValueError(f"Malformed or corrupted PDF: {e}")
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing excessive whitespace and special characters
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # Remove zero-width spaces and other invisible characters
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
        return text.strip()
    
    def _validate_parsed_data(self, data: dict) -> dict:
        """Validate and clean parsed data
        
        Args:
            data: Parsed resume data
            
        Returns:
            Validated and cleaned data
        """
        # Ensure all required top-level fields exist
        required_fields = ['basics', 'summary', 'work', 'education', 'skills', 
                          'certifications', 'achievements', 'languages']
        
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing field '{field}' in parsed data, adding empty value")
                if field in ['work', 'education', 'certifications', 'achievements', 'languages']:
                    data[field] = []
                elif field == 'skills':
                    data[field] = {'technical': [], 'soft': []}
                elif field == 'basics':
                    data[field] = {
                        'name': '', 'email': '', 'phone': '',
                        'location': {'city': '', 'countryCode': ''}
                    }
                else:
                    data[field] = ''
        
        # Clean empty strings from arrays
        for field in ['work', 'education', 'certifications', 'achievements', 'languages']:
            if isinstance(data.get(field), list):
                data[field] = [item for item in data[field] if item]
        
        return data
    
    def parse_resume(self, pdf_path: str, save_output: bool = False, output_path: str = None) -> dict:
        """Parse a resume PDF and convert it to structured JSON with comprehensive error handling
        
        Args:
            pdf_path: Path to the resume PDF file
            save_output: Whether to save the output to a JSON file (default: False)
            output_path: Path to save the JSON file (only used if save_output=True)
            
        Returns:
            Parsed resume data as dictionary
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF is invalid or parsing fails
            Exception: For other unexpected errors
        """
        try:
            logger.info(f"Starting to parse resume: {pdf_path}")
            resume_text = self.read_pdf(pdf_path)
        except Exception as e:
            logger.error(f"Failed to read PDF: {e}")
            raise
        
        # Few-shot example with actual sample data
        sample_resume_text = """John Doe
San Francisco
CA
USA
☎ +1 555 123 4567
✉ john.doe@techcorp.com

Professional Summary
Experienced AI Engineer with 3+ years of expertise in developing scalable machine learning solutions and real-time analytics systems. Proven track record of delivering high-impact projects that improve operational efficiency.

Experience
2020–2023 Senior AI Engineer , Tech Corp, San Francisco, CA
Led development of a real-time predictive analytics engine used by 50+ clients, increasing internal
efficiency by 30%. Focused on deep learning models and scalable cloud infrastructure.

Education
2018–2020 Master of Science in Computer Science (Artificial Intelligence) , University of Tech

Awards
2021 Innovation of the Year – Tech Corp Internal Summit
Recognized for creating a proprietary AI-driven fraud detection system

Certifications
2022 Certified Cloud Professional (GCP) – Google Cloud

Skills
Technical Skills: Python · TypeScript· Go · PyTorch · TensorFlow· Keras · Scikit-learn · Docker · Kubernetes
Soft Skills: Team Leadership · Problem Solving · Communication · Project Management

Languages
English (Native)
Spanish (Professional Working Proficiency)"""

        sample_json_output = {
            "basics": {
                "name": "John Doe",
                "email": "john.doe@techcorp.com",
                "phone": "+1-555-123-4567",
                "location": {
                    "city": "San Francisco",
                    "countryCode": "US"
                }
            },
            "summary": "Experienced AI Engineer with 3+ years of expertise in developing scalable machine learning solutions and real-time analytics systems. Proven track record of delivering high-impact projects that improve operational efficiency.",
            "work": [
                {
                    "company": "Tech Corp",
                    "position": "Senior AI Engineer",
                    "startDate": "2020-01",
                    "endDate": "2023-12",
                    "summary": "Led development of a real-time predictive analytics engine used by over 50 clients, increasing internal efficiency by 30%. Focused on deep learning models and scalable cloud infrastructure."
                }
            ],
            "education": [
                {
                    "institution": "University of Tech",
                    "area": "Computer Science (Artificial Intelligence)",
                    "studyType": "Master of Science",
                    "startDate": "2018-09-01",
                    "endDate": "2020-06-01"
                }
            ],
            "skills": {
                "technical": [
                    "Python",
                    "TypeScript",
                    "Go",
                    "PyTorch",
                    "TensorFlow",
                    "Keras",
                    "Scikit-learn",
                    "Docker",
                    "Kubernetes"
                ],
                "soft": [
                    "Team Leadership",
                    "Problem Solving",
                    "Communication",
                    "Project Management"
                ]
            },
            "certifications": [
                {
                    "name": "Certified Cloud Professional (GCP)",
                    "issuer": "Google Cloud",
                    "date": "2022-03-01"
                }
            ],
            "achievements": [
                {
                    "title": "Innovation of the Year",
                    "awarder": "Tech Corp Internal Summit",
                    "date": "2021-06-15",
                    "description": "Recognized for the creation of the proprietary AI-driven fraud detection system."
                }
            ],
            "languages": [
                {
                    "language": "English",
                    "fluency": "Native"
                },
                {
                    "language": "Spanish",
                    "fluency": "Professional Working Proficiency"
                }
            ]
        }
        
        # Create prompt template with few-shot example
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume parser. Extract information from the resume text and structure it according to the JSON schema provided.

CRITICAL RULES - YOU MUST FOLLOW THESE STRICTLY:
1. ONLY extract information that is EXPLICITLY present in the resume text
2. DO NOT invent, generate, or fabricate ANY information
3. DO NOT use placeholder values like "Company Name", "University Name", etc.
4. If a field value is not found in the resume, use an empty string "" or empty array []
5. For dates, convert to the specified format (YYYY-MM-DD or YYYY-MM) based on available information
6. Maintain the hierarchical structure as shown in the schema
7. Handle various CV layouts: traditional, modern, multi-column, tables, and bullet points
8. Extract information even if formatting is inconsistent or unconventional
9. Infer company names from context if explicitly mentioned near job positions
10. For contact information, look in headers, footers, and throughout the document

FEW-SHOT EXAMPLE:
Given this resume text:
{sample_resume_text}

The correct output is:
{sample_json_output}

Notice how ACTUAL values are extracted:
- Name: "John Doe" (not "Full Name")
- Company: "Tech Corp" (not "Company Name")
- Institution: "University of Tech" (not "University Name")
- Position: "Senior AI Engineer" (not "Job Title")
- Skills are categorized into technical and soft skills
- Languages include fluency levels

{format_instructions}"""),
            ("user", "Now parse this resume text and extract ONLY the actual values present:\n\n{resume_text}")
        ])
        
        # Create chain
        chain = prompt | self.llm | self.parser
        
        # Parse resume with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Parsing resume (attempt {attempt + 1}/{max_retries})")
                result = chain.invoke({
                    "resume_text": resume_text,
                    "format_instructions": self.parser.get_format_instructions(),
                    "sample_resume_text": sample_resume_text,
                    "sample_json_output": json.dumps(sample_json_output, indent=2)
                })
                
                # Validate parsed data
                result = self._validate_parsed_data(result)
                logger.info("Resume parsed successfully")
                
                # Save to file if requested
                if save_output and output_path:
                    self.save_to_json(result, output_path)
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to parse resume after {max_retries} attempts: Invalid JSON output")
            except Exception as e:
                logger.error(f"Parsing error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to parse resume: {e}")
        
        # Fallback: return minimal structure if all retries fail
        logger.warning("All parsing attempts failed, returning minimal structure")
        empty_data = self._get_empty_resume_structure()
        
        if save_output and output_path:
            self.save_to_json(empty_data, output_path)
        
        return empty_data
    
    def _get_empty_resume_structure(self) -> dict:
        """Return an empty resume structure as fallback
        
        Returns:
            Empty resume structure dictionary
        """
        return {
            "basics": {
                "name": "",
                "email": "",
                "phone": "",
                "location": {"city": "", "countryCode": ""}
            },
            "summary": "",
            "work": [],
            "education": [],
            "skills": {"technical": [], "soft": []},
            "certifications": [],
            "achievements": [],
            "languages": []
        }
    
    def save_to_json(self, data: dict, output_path: str) -> None:
        """Save parsed resume data to a JSON file with error handling
        
        Args:
            data: Parsed resume data
            output_path: Path to save the JSON file
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved parsed resume to: {output_path}")
        except IOError as e:
            logger.error(f"Failed to save JSON file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while saving JSON: {e}")
            raise


def main():
    """Main function demonstrating usage with comprehensive error handling"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Parse resume PDF to structured JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s resume.pdf
  %(prog)s resume.pdf -o output.json
  %(prog)s resume.pdf --provider gemini
  %(prog)s resume.pdf -o output.json -v
        """
    )
    parser.add_argument("pdf_path", help="Path to the resume PDF file")
    parser.add_argument("-o", "--output", help="Output JSON file path", 
                       default="data/outputs/parsed_resume.json")
    parser.add_argument("-p", "--provider", choices=["azure", "gemini"], 
                       default="azure", help="LLM provider to use (default: azure)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize parser
        print(f"Initializing parser with provider: {args.provider}")
        resume_parser = ResumeParser(provider=args.provider)
        
        # Parse resume with save option
        print(f"Parsing resume from: {args.pdf_path}")
        parsed_data = resume_parser.parse_resume(
            args.pdf_path,
            save_output=True,
            output_path=args.output
        )
        print(f"✓ Resume parsed successfully and saved to: {args.output}")
        
        # Print preview
        print("\n" + "="*50)
        print("Parsed Data Preview:")
        print("="*50)
        preview = json.dumps(parsed_data, indent=2)
        if len(preview) > 500:
            print(preview[:500] + "\n...\n(truncated)")
        else:
            print(preview)
        
        # Print statistics
        print("\n" + "="*50)
        print("Extraction Statistics:")
        print("="*50)
        print(f"Name: {'✓' if parsed_data.get('basics', {}).get('name') else '✗'}")
        print(f"Email: {'✓' if parsed_data.get('basics', {}).get('email') else '✗'}")
        print(f"Work Experience: {len(parsed_data.get('work', []))} entries")
        print(f"Education: {len(parsed_data.get('education', []))} entries")
        print(f"Technical Skills: {len(parsed_data.get('skills', {}).get('technical', []))} items")
        print(f"Certifications: {len(parsed_data.get('certifications', []))} entries")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1
    except EnvironmentError as e:
        print(f"✗ Configuration Error: {e}", file=sys.stderr)
        print("\nPlease ensure all required environment variables are set in .env file", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Unexpected Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
