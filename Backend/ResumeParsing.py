import torch
import io
import re
import threading
import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification, pipeline
from PyPDF2 import PdfReader

class ResumeParser:

    COMBINED_PATTERN = re.compile(
        r"(?P<name>^\s*([A-Z][a-z]+(?:\s+(?:[A-Z]\.|[A-Z][a-z]+))*)(?=\s+(?:Contact|Address)\b))"
        r"|(?P<email>[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
        r"|(?P<phone>\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b)"
        r"|(?P<address>\b\d{3,5}\s+\d{1,4}(?:st|nd|rd|th)\s+\w+(?:\s+\w+)?\b)"
        r"|(?P<year>(?:19|20)\d{2})",
        re.IGNORECASE | re.MULTILINE,
    )

    EXP_SECTION_PATTERN = re.compile(
        r"(Tech/Coding Experience:.*)", re.DOTALL | re.IGNORECASE
    )
    ORGS_PATTERN = re.compile(r"(?:at|for)\s+([A-Z][a-zA-Z0-9&.,\s]+)")

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls, model_name="dslim/bert-base-NER", device_preference=None):
        # device_preference: None → auto; -1 → CPU; 0 → first GPU
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(
                    model_name=model_name, device_preference=device_preference
                )
            return cls._instance

    def __init__(self, model_name="dslim/bert-base-NER", device_preference=None):

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        # --- Device preference ---
        if device_preference is not None:
            self._device = device_preference
        else:
            self._device = 0 if torch.cuda.is_available() else -1

        # --- Target device string ---
        target_device = (
            "cuda" if self._device != -1 and torch.cuda.is_available() else "cpu"
        )

        # --- Tokenizer ---
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Disable gradients globally (inference only)
        torch.set_grad_enabled(False)

        # --- Load model directly on target device (no meta tensors) ---
        self.model = BertForTokenClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"": target_device},  # ensures model is fully on chosen device
            low_cpu_mem_usage=False,  # force full load
            force_download=False,
            ignore_mismatched_sizes=True,
        )

        self.model.eval()

        # --- Final sanity check ---
        if any(p.is_meta for p in self.model.parameters()):
            raise RuntimeError(
                "Model still contains meta tensors even after safe load!"
            )

        # --- Pipeline ---
        self.nlp = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            framework="pt",
        )

    def _is_meta(self):
        try:
            return next(self.model.parameters()).is_meta
        except StopIteration:
            return True

    def _ensure_ready(self):
        if self._is_meta():
            # Hard rebuild if something left us in a meta state
            type(self)._instance = None
            rebuilt = type(self).get(device_preference=self._device)
            # swap internals so existing references continue to work
            self.__dict__.update(rebuilt.__dict__)

    def parse(self, resume_text: str):
        """NLP-based entity parsing."""
        self._ensure_ready()
        if not resume_text or not resume_text.strip():
            return {}
        with torch.inference_mode():
            entities = self.nlp(resume_text)
        parsed = {}
        for ent in entities:
            parsed.setdefault(ent["entity_group"], []).append(ent["word"])
        for label, words in parsed.items():
            if label.lower() != "skills":
                parsed[label] = [" ".join(words)]
        return parsed

    def _load_text(self, resume_bytes_or_path):
        # stub; reuse your own implementation
        return resume_bytes_or_path if isinstance(resume_bytes_or_path, str) else ""

    @staticmethod
    def get_resume_text(file, resume_text=""):
        if hasattr(file, "type") and file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                resume_text += page.extract_text() or ""
        else:
            if hasattr(file, "read"):
                content = file.read()
                resume_text = (
                    content.decode("utf-8") if isinstance(content, bytes) else content
                )
            else:
                with open(file, "r", encoding="utf-8") as f:
                    resume_text = f.read()

        return resume_text

    def get_parsed_data(self, resume_text):
        parsed = {
            "Name": None,
            "Email Address": None,
            "Mobile Number": None,
            "Address": None,
            "Skills": [],
            "Degree": None,
            "College Name": None,
            "Graduation Year": None,
            "Company Name": None,
        }

        nlp_parsed = self.parse(resume_text)

        label_map = {
            "PERSON": "Name",
            "EMAIL_ADDRESS": "Email Address",
            "PHONE_NUMBER": "Mobile Number",
            "ADDRESS": "Address",
            "SKILL": "Skills",
        }

        for label, key in label_map.items():
            if label in nlp_parsed and not parsed[key]:
                if key == "Skills":
                    parsed[key] = list(dict.fromkeys(nlp_parsed[label]))
                else:
                    parsed[key] = nlp_parsed[label][0]

        return parsed

    @staticmethod
    def parse_skills(resume_text):
        skills_match = re.search(
            r"Skills\s*[:\-]?\s*(.*?)(?:\n[A-Z][a-z]+:|\n\n|\Z)", resume_text, re.DOTALL
        )
        if skills_match:
            skills_text = skills_match.group(1)
            skills_text = re.sub(r"[•●]", ",", skills_text)
            skills_text = skills_text.replace("\n", " ")
            skills_text = re.sub(r"\s+", " ", skills_text)
            raw_skills = [s.strip() for s in skills_text.split(",") if s.strip()]
            return list(dict.fromkeys(raw_skills))
        return []

    @staticmethod
    def get_designations(resume_text):
        designations = []
        edu_section = re.search(
            r"Education:(.*?)(?:Tech/Coding Experience:|$)",
            resume_text,
            re.DOTALL | re.IGNORECASE,
        )
        if edu_section:
            degree_match = re.search(
                r"(?<=\bmajoring in )([^.]*\.)", edu_section.group(1), re.IGNORECASE
            )
            if degree_match:
                designations.append(degree_match.group(0).replace("Majors", "").strip())
            college_match = re.search(
                r"(University of [A-Za-z]+(?:[-\s][A-Za-z]+)*)(?=\s(?:majoring|studying|with|,|\.))",
                edu_section.group(1),
                re.IGNORECASE,
            )
            if college_match:
                designations.append(college_match.group(0).strip())
        return designations

    def generate_formatted_resumes(self, parsed_data):
        """
        Generate the same resume in multiple clean and fancy formats.
        Returns a dict of format_name: resume_text.
        """

        def safe_str(value, default=""):
            """Return string of value, or default if value is None."""
            return default if value is None else str(value)

        if parsed_data is None:
            return {}

        st.write("Parsed Resume Data:", parsed_data)

        name = safe_str(parsed_data.get("Name"))
        email = safe_str(parsed_data.get("Email Address"))
        phone = safe_str(parsed_data.get("Mobile Number"))
        address = safe_str(parsed_data.get("Address"))
        degree = safe_str(parsed_data.get("Degree"))
        college = safe_str(parsed_data.get("College Name"))
        grad_year = safe_str(parsed_data.get("Graduation Year"))
        company = safe_str(parsed_data.get("Company Name"))

        # Handle skills safely
        raw_skills = parsed_data.get("Skills")
        if raw_skills is None:
            skills = ""
        elif isinstance(raw_skills, (list, tuple)):
            # filter out None/empty entries, convert each to string
            skills = ", ".join([safe_str(s) for s in raw_skills if s not in (None, "")])
        else:
            skills = safe_str(raw_skills)

        # Modern format
        modern = f"""\
    {name}
    {address}
    {phone} | {email}

    SUMMARY
    --------
    Results-driven professional with proven experience.

    EDUCATION
    ---------
    {degree} ({grad_year})
    {college}

    EXPERIENCE
    ----------
    {company}

    SKILLS
    ------
    {skills}
    """

        # Boxed format
        boxed = (
            f"+{'-'*50}+\n"
            f"| {name.center(48)} |\n"
            f"+{'-'*50}+\n"
            f"| Address: {address.ljust(38)} |\n"
            f"| Phone:   {phone.ljust(38)} |\n"
            f"| Email:   {email.ljust(38)} |\n"
            f"+{'-'*50}+\n"
            f"| EDUCATION: {degree} ({grad_year}){' '*(27 - len(degree + grad_year))}|\n"
            f"|           {college.ljust(32)} |\n"
            f"+{'-'*50}+\n"
            f"| EXPERIENCE: {company.ljust(34)} |\n"
            f"+{'-'*50}+\n"
            f"| SKILLS: {skills.ljust(39)} |\n"
            f"+{'-'*50}+\n"
        )

        # Minimalist format
        minimalist = f"""\
    {name} | {email} | {phone}
    {address}

    Education: {degree} ({grad_year}), {college}
    Experience: {company}
    Skills: {skills}
    """

        return {
            "modern": modern,
            "boxed": boxed,
            "minimalist": minimalist,
        }

    # O(n) time complexity
    def parse_resume_file(self, file):
        resume_text = self.get_resume_text(file)
        parsed = self.get_parsed_data(resume_text)

        # Normalize whitespace
        resume_text = re.sub(r"\s+", " ", resume_text).strip()

        matches = {
            "name": None,
            "email": None,
            "phone": None,
            "address": None,
            "years": [],
            "exp_section": None,
            "orgs": [],
        }

        # Single-pass scan for name/email/phone/address/years
        for m in self.COMBINED_PATTERN.finditer(resume_text):
            if m.group("name") and not matches["name"]:
                matches["name"] = m.group("name").strip()
            elif m.group("email") and not matches["email"]:
                matches["email"] = m.group("email")
            elif m.group("phone") and not matches["phone"]:
                matches["phone"] = m.group("phone")
            elif m.group("address") and not matches["address"]:
                matches["address"] = m.group("address").strip()
            elif m.group("year"):
                matches["years"].append(m.group("year"))

        # Experience + organizations
        exp_section_match = self.EXP_SECTION_PATTERN.search(resume_text)
        if exp_section_match:
            matches["exp_section"] = exp_section_match.group(1)
            matches["orgs"] = self.ORGS_PATTERN.findall(matches["exp_section"])

        # Assign parsed fields
        if matches["name"]:
            parsed["Name"] = matches["name"]
        if matches["email"]:
            parsed["Email Address"] = matches["email"]
        if matches["phone"]:
            parsed["Mobile Number"] = matches["phone"]
        if matches["address"]:
            parsed["Address"] = matches["address"]

        # Skills
        parsed["Skills"] = self.parse_skills(resume_text)

        # Designation + College
        designations = self.get_designations(resume_text)
        if designations:
            if len(designations) > 0:
                parsed["Degree"] = designations[0]
            if len(designations) > 1:
                parsed["College Name"] = designations[1]

        # Graduation Year: last/latest
        if matches["years"]:
            parsed["Graduation Year"] = max(matches["years"])

        # Company Name: from experience section
        if matches["orgs"]:
            parsed["Company Name"] = matches["orgs"][0].strip()

        return parsed

    def extract_skills(self, text) -> list:
        return text.get("Skills", []) if isinstance(text, dict) else []

    def get_job_skills(self, job_title: str) -> list:
        df = pd.read_csv("jobss.csv")
        subset = df[df["Job Title"].str.contains(job_title, case=False, na=False)]

        skills = set()
        for ks in subset["Key Skills"]:
            if isinstance(ks, str):
                skills.update([skill.strip() for skill in ks.split(",")])
        skills_list = []
        for skill in skills:
            if isinstance(skill, str):
                parts = [skill.strip() for skill in skill.split("|") if skill.strip()]
                skills_list.extend(parts)
        return list(set(skills_list))

    def match_skills_to_job(self, resume_data, job_description: str) -> dict:

        if resume_data is None:
            return {
                "matched_skills": [],
                "missing_skills": [],
                "resume_skills": [],
                "job_skills": [],
            }

        resume_skills = set(self.extract_skills(resume_data))
        resume_skills = {skill.casefold() for skill in resume_skills}
        job_skills = set(self.get_job_skills(job_description))
        job_skills = {skill.casefold() for skill in job_skills}

        matched_skills = resume_skills.intersection(job_skills)
        missing_skills = job_skills.difference(resume_skills)
        return {
            "matched_skills": list(matched_skills),
            "missing_skills": list(missing_skills),
            "resume_skills": list(resume_skills),
            "job_skills": list(job_skills),
        }
