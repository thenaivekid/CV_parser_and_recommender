"""
Resume Parser using LangChain with Azure OpenAI or Google Gemini
Reads PDF resumes and converts them to structured JSON format
"""

import os
import json
from typing import Literal
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class Location(BaseModel):
    """Location information"""
    city: str = Field(description="City name")
    countryCode: str = Field(description="Two-letter country code")


class Basics(BaseModel):
    """Basic information about the person"""
    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")
    location: Location = Field(description="Location information")


class WorkExperience(BaseModel):
    """Work experience entry"""
    company: str = Field(description="Company name")
    position: str = Field(description="Job position/title")
    startDate: str = Field(description="Start date in YYYY-MM format")
    endDate: str = Field(description="End date in YYYY-MM format or 'Present'")
    summary: str = Field(description="Summary of responsibilities and achievements")


class VolunteerExperience(BaseModel):
    """Volunteer experience entry"""
    organization: str = Field(description="Organization name")
    position: str = Field(description="Role/position")
    url: str = Field(default="", description="Organization URL")
    startDate: str = Field(description="Start date in YYYY-MM-DD format")
    endDate: str = Field(description="End date in YYYY-MM-DD format")
    summary: str = Field(description="Summary of contributions")
    highlights: list[str] = Field(default=[], description="Key achievements")


class Education(BaseModel):
    """Education entry"""
    institution: str = Field(description="Institution name")
    url: str = Field(default="", description="Institution URL")
    area: str = Field(description="Field of study")
    studyType: str = Field(description="Degree type (e.g., Bachelor of Science)")
    startDate: str = Field(description="Start date in YYYY-MM-DD format")
    endDate: str = Field(description="End date in YYYY-MM-DD format")
    score: str = Field(default="", description="GPA or score")
    courses: list[str] = Field(default=[], description="Relevant courses")


class Award(BaseModel):
    """Award entry"""
    title: str = Field(description="Award title")
    date: str = Field(description="Award date in YYYY-MM-DD format")
    awarder: str = Field(description="Organization that gave the award")
    summary: str = Field(description="Award description")


class Certificate(BaseModel):
    """Certificate entry"""
    name: str = Field(description="Certificate name")
    date: str = Field(description="Issue date in YYYY-MM-DD format")
    issuer: str = Field(description="Issuing organization")
    url: str = Field(default="", description="Certificate URL")


class Publication(BaseModel):
    """Publication entry"""
    name: str = Field(description="Publication title")
    publisher: str = Field(description="Publisher name")
    releaseDate: str = Field(description="Release date in YYYY-MM-DD format")
    url: str = Field(default="", description="Publication URL")
    summary: str = Field(description="Publication summary")


class Skill(BaseModel):
    """Skill entry"""
    name: str = Field(description="Skill category name")
    level: str = Field(description="Proficiency level")
    keywords: list[str] = Field(description="List of specific skills")


class Project(BaseModel):
    """Project entry"""
    name: str = Field(description="Project name")
    startDate: str = Field(description="Start date in YYYY-MM-DD format")
    endDate: str = Field(description="End date in YYYY-MM-DD format")
    description: str = Field(description="Project description")
    highlights: list[str] = Field(default=[], description="Key achievements")
    url: str = Field(default="", description="Project URL")


class ResumeData(BaseModel):
    """Complete resume data structure"""
    basics: Basics = Field(description="Basic personal information")
    work: list[WorkExperience] = Field(default=[], description="Work experience")
    volunteer: list[VolunteerExperience] = Field(default=[], description="Volunteer experience")
    education: list[Education] = Field(default=[], description="Education history")
    awards: list[Award] = Field(default=[], description="Awards and honors")
    certificates: list[Certificate] = Field(default=[], description="Certifications")
    publications: list[Publication] = Field(default=[], description="Publications")
    skills: list[Skill] = Field(default=[], description="Skills")
    projects: list[Project] = Field(default=[], description="Projects")


class ResumeParser:
    """Parser for converting PDF resumes to structured JSON"""
    
    def __init__(self, provider: Literal["azure", "gemini"] = "azure"):
        """
        Initialize the resume parser
        
        Args:
            provider: LLM provider to use ("azure" for Azure OpenAI or "gemini" for Google Gemini)
        """
        self.provider = provider
        self.llm = self._initialize_llm()
        self.parser = JsonOutputParser(pydantic_object=ResumeData)
        
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on provider"""
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
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def read_pdf(self, pdf_path: str) -> str:
        """
        Read text content from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def parse_resume(self, pdf_path: str) -> dict:
        """
        Parse a resume PDF and convert it to structured JSON
        
        Args:
            pdf_path: Path to the resume PDF file
            
        Returns:
            Parsed resume data as dictionary
        """
        # Read PDF content
        resume_text = self.read_pdf(pdf_path)
        
        # Few-shot example with actual sample data
        sample_resume_text = """John Doe
San Francisco
CA
USA
Ȳ +1 555 123 4567
a john.doe@techcorp.com
Experience
2020–2023 Senior AI Engineer , T ech Corp, San Francisco, CA
Led development of a real-time predictive analytics engine used by 50+ clients, increasing internal
eﬀiciency by 30%. Focused on deep learning models and scalable cloud infrastructure.
Projects
2023 Financial Data Predictor , Personal Project
f Developed an LSTM model to predict short-term stock volatility (65% accuracy on test data)
f Deployed using Docker and Kubernetes
f GitHub: https://github.com/johndoe/stock-predictor
Education
2018–2020 Master of Science in Computer Science (Artificial Intelligence) , University of T ech,
3.9/4.0
Relevant coursework: Advanced Neural Networks · Distributed Systems and Big Data
Open Source & Volunteer
2018–2020 Lead Contributor, Open Source Initiative – Project XYZ
f Managed a team of 5 developers on a popular data visualization library
f Mentored two junior contributors on best practices
f Successfully migrated the entire codebase to T ypeScript
https://opensource.org/project-xyz
Publications
2023 "Scaling Model T raining with Serverless Functions"
Journal of Ma-
chine Learning
Systems
https://journal-
mls.com/john-doe-
paper
Awards
2021 Innovation of the Y ear – T ech Corp Internal Summit
Recognized for cre-
ating a proprietary
AI-driven fraud de-
tection system
Certifications
2022 Certified Cloud Professional (GCP) – Google Cloud
Skills
Programming
Languages
Python · T ypeScript· Go
AI/ML
Frameworks
PyT orch (Master)· T ensorFlow· Keras · Scikit-learn
Cloud &
Infrastructure
GCP (Certified) · Docker · Kubernetes · Serverless"""

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
            "work": [
                {
                    "company": "Tech Corp",
                    "position": "Senior AI Engineer",
                    "startDate": "2020-01",
                    "endDate": "2023-12",
                    "summary": "Led development of a real-time predictive analytics engine used by over 50 clients, increasing internal efficiency by 30%. Focused on deep learning models and scalable cloud infrastructure."
                }
            ],
            "volunteer": [
                {
                    "organization": "Open Source Initiative",
                    "position": "Lead Contributor",
                    "url": "https://opensource.org/project-xyz",
                    "startDate": "2018-05-01",
                    "endDate": "2020-01-01",
                    "summary": "Managed a team of 5 developers contributing to Project XYZ, a popular data visualization library.",
                    "highlights": [
                        "Mentored two junior contributors on best practices",
                        "Successfully migrated the codebase to TypeScript"
                    ]
                }
            ],
            "education": [
                {
                    "institution": "University of Tech",
                    "url": "https://u-tech.edu/",
                    "area": "Computer Science (Artificial Intelligence)",
                    "studyType": "Master of Science",
                    "startDate": "2018-09-01",
                    "endDate": "2020-06-01",
                    "score": "3.9/4.0",
                    "courses": [
                        "Advanced Neural Networks (CS501)",
                        "Distributed Systems and Big Data (CS502)"
                    ]
                }
            ],
            "awards": [
                {
                    "title": "Innovation of the Year",
                    "date": "2021-06-15",
                    "awarder": "Tech Corp Internal Summit",
                    "summary": "Recognized for the creation of the proprietary AI-driven fraud detection system."
                }
            ],
            "certificates": [
                {
                    "name": "Certified Cloud Professional (GCP)",
                    "date": "2022-03-01",
                    "issuer": "Google Cloud",
                    "url": "https://certs.google.com/john-doe-gcp-cert"
                }
            ],
            "publications": [
                {
                    "name": "Scaling Model Training with Serverless Functions",
                    "publisher": "Journal of Machine Learning Systems",
                    "releaseDate": "2023-08-01",
                    "url": "https://journal-mls.com/john-doe-paper",
                    "summary": "A detailed analysis of infrastructure techniques to reduce the cost and time of large-scale model retraining."
                }
            ],
            "skills": [
                {
                    "name": "Programming Languages",
                    "level": "Expert",
                    "keywords": ["Python", "TypeScript", "Go"]
                },
                {
                    "name": "AI/ML Frameworks",
                    "level": "Master",
                    "keywords": ["PyTorch", "TensorFlow", "Keras", "Scikit-learn"]
                }
            ],
            "projects": [
                {
                    "name": "Financial Data Predictor",
                    "startDate": "2023-01-01",
                    "endDate": "2023-06-30",
                    "description": "A personal side project developing an LSTM model to predict short-term stock volatility.",
                    "highlights": [
                        "Achieved 65% prediction accuracy on test data",
                        "Deployed using Docker and Kubernetes"
                    ],
                    "url": "https://github.com/johndoe/stock-predictor"
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

{format_instructions}"""),
            ("user", "Now parse this resume text and extract ONLY the actual values present:\n\n{resume_text}")
        ])
        
        # Create chain
        chain = prompt | self.llm | self.parser
        
        # Parse resume
        result = chain.invoke({
            "resume_text": resume_text,
            "format_instructions": self.parser.get_format_instructions(),
            "sample_resume_text": sample_resume_text,
            "sample_json_output": json.dumps(sample_json_output, indent=2)
        })
        
        return result
    
    def save_to_json(self, data: dict, output_path: str):
        """
        Save parsed resume data to a JSON file
        
        Args:
            data: Parsed resume data
            output_path: Path to save the JSON file
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    """Main function demonstrating usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse resume PDF to structured JSON")
    parser.add_argument("pdf_path", help="Path to the resume PDF file")
    parser.add_argument("-o", "--output", help="Output JSON file path", 
                       default="data/outputs/parsed_resume.json")
    parser.add_argument("-p", "--provider", choices=["azure", "gemini"], 
                       default="azure", help="LLM provider to use")
    
    args = parser.parse_args()
    
    # Initialize parser
    resume_parser = ResumeParser(provider=args.provider)
    
    # Parse resume
    print(f"Parsing resume from: {args.pdf_path}")
    print(f"Using provider: {args.provider}")
    parsed_data = resume_parser.parse_resume(args.pdf_path)
    
    # Save to JSON
    resume_parser.save_to_json(parsed_data, args.output)
    print(f"Resume parsed successfully and saved to: {args.output}")
    
    # Print preview
    print("\nParsed Data Preview:")
    print(json.dumps(parsed_data, indent=2)[:500] + "...")


if __name__ == "__main__":
    main()
