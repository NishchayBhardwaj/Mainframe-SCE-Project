# E-Governance Public Grievance Analyzer

A hybrid AI + Mainframe system to automate public grievance analysis using Natural Language Processing (NLP), legacy system integration, and modern web interfaces.

## Overview

Public grievance systems often face challenges like manual triaging, data fragmentation, and delayed resolutions. This project presents a smart, integrated solution that leverages modern AI techniques (Groq Llama-3.3-70b) and connects seamlessly with legacy mainframe databases (VSAM KSDS) to automate the categorization, prioritization, and analysis of grievances.

## Key Features

- **AI-based Complaint Categorization** using Groq Llama-3.3-70b
- **Legacy Mainframe Integration** using Paramiko to extract data from VSAM KSDS
- **Document Parsing** with PyMuPDF to handle scanned PDFs and attachments
- **Secure Admin Portal** using Streamlit for submission, classification, and analytics
- **Real-time Grievance Monitoring** with filters, priority tags, and status dashboards

## System Architecture

- **Frontend**: Streamlit
- **Backend**: Python 3, Groq Llama-3.3-70b
- **Mainframe Integration**: Paramiko (SSH), JCL for batch processing
- **Database**: IBM z/OS VSAM KSDS
- **Document Parser**: PyMuPDF (fitz)

## Methodology

1. **Data Extraction**: Secure SSH connection using Paramiko to pull grievance data from VSAM KSDS.
2. **Preprocessing**: Convert EBCDIC to UTF-8 and parse attached documents.
3. **AI Processing**: Use LLM to:
   - Classify grievance category
   - Predict priority level
   - Extract keywords
4. **Visualization**: Admin dashboard for monitoring complaint trends and response status.

## Setup Instructions

### Requirements

- Python 3.9+
- Streamlit
- Paramiko
- PyMuPDF
- Groq SDK (or compatible LLM interface)

### Installation

```bash
git clone https://github.com/NishchayBhardwaj/Mainframe-SCE-Project.git
cd Mainframe-SCE-Project
pip install -r requirements.txt
streamlit run app.py
