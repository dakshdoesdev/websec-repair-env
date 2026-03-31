# Web Security Review Playbook

This markdown file is the local plain-text research input for the WebSec Repair
training pipeline. It distills the longer security environment blueprint into
structured policy signals that can be converted into task-grounded SFT examples.

## Closed-Loop Workflow

Every secure repair follows the same loop:

1. `inspect`
2. `classify`
3. `apply_patch`
4. `verify`
5. `submit`

The model should avoid skipping directly to a patch without evidence. A fix is
only considered good after exploit blocking and functionality preservation are
both verified.

## Taxonomy

- Use **OWASP Top 10 2021** categories for high-level reporting.
- Use **MITRE CWE** identifiers for precise vulnerability labels.
- Record **CVSS v3.1** score and vector when the risk is known.

Important mappings for this project:

- SQL injection -> `A03: Injection`, `CWE-89`
- XSS -> `A03: Injection`, `CWE-79`
- Broken auth / missing authorization -> `A01: Broken Access Control`, `CWE-862`
- IDOR -> `A01: Broken Access Control`, `CWE-639`
- Path traversal -> `A01: Broken Access Control`, `CWE-22`
- SSRF -> `A10: Server-Side Request Forgery`, `CWE-918`
- Insecure deserialization -> `A08: Software and Data Integrity Failures`, `CWE-502`
- Command injection -> `A03: Injection`, `CWE-78`
- XPath injection -> `A03: Injection`, `CWE-643`
- Template injection -> `A03: Injection`, `CWE-1336`
- CSRF -> `CWE-352`
- Hard-coded secret -> `CWE-798`
- Vulnerable dependency -> `A06: Vulnerable and Outdated Components`, `CWE-1104`

## Tooling

Preferred verification and evidence sources:

- **Semgrep** for SAST findings and CWE / OWASP mapping
- **Trivy** for dependency, secret, and misconfiguration scanning
- **OWASP ZAP** automation for runtime regression in sandboxed lab mode

## Verification Contract

Do not accept “fix suggested” as success. Require:

- exploit probe blocked
- functionality preserved
- rescans cleaned or improved

Use:

- deterministic exploit probes in the MVP
- unit or integration tests for preserved behavior
- Semgrep rescans for code-pattern validation
- Trivy rescans for secrets, dependencies, and misconfigurations
- ZAP automation in real lab mode only and only for permitted targets

## Reward Philosophy

Reward secure, minimal, verifiable fixes. Penalize:

- weak filtering
- UI-only mitigations
- disabling core functionality when a safe fix exists
- repeated rescans without new evidence

## Vulnerability Families

This training corpus should cover at least:

- sql injection
- xss
- broken auth
- idor
- path traversal
- ssrf
- command injection
- insecure deserialization
- template injection
- csrf
- hard-coded credentials
- vulnerable dependencies
- weak password hashing
- dangerous file upload

## Serving Constraint

The trained model must still emit a compact environment action that works with
the current OpenEnv loop, but it can also return richer review metadata.
