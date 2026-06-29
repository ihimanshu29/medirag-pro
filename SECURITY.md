# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ Active |

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email: security@yourdomain.com  
Response time: 48 hours  
Patch target: 7 days for critical, 30 days for moderate

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive acknowledgement within 48 hours and a resolution timeline within 5 business days.

---

## Known Security Considerations

### LLM Output
This system uses a large language model (Groq Llama 3.3 70B) which may produce unexpected or incorrect outputs. All responses include a medical disclaimer. Do not use this system for actual clinical decision-making.

### Emergency Detection
The emergency detection layer uses keyword matching and regex patterns. It is not a substitute for professional crisis services. If you are experiencing a medical emergency, call your local emergency number.

### Data Privacy
- User queries are logged via structlog — do not ingest documents containing personally identifiable information (PII)
- Conversation history is stored in PostgreSQL — secure your database accordingly
- The Groq API processes your queries — review [Groq's privacy policy](https://groq.com/privacy-policy/) before use

### BM25 Index
The BM25 index is persisted as a Python pickle file (`data/bm25_index.pkl`). Only load this file from sources you trust. The application validates the source file at load time.

### File Upload
- Only PDF and DOCX files are accepted
- Maximum file size: 50MB
- Filenames are sanitized to prevent path traversal

### API Keys
- API keys are loaded from environment variables only — never committed to source control
- The `.env` file is in `.gitignore`
- Use the `.env.example` template which contains no real credentials

### Rate Limiting
The `/chat` endpoint is rate-limited to 20 requests/minute per IP. The `/ingest` endpoint accepts up to 5 requests/minute per IP.

---

## Dependency Security

Dependencies are scanned for known vulnerabilities using `pip-audit` in CI on every push to main. To scan locally:

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

To check for outdated packages:
```bash
pip list --outdated
```
