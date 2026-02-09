Perfect 👍
Below is a **clean, enterprise-ready Design Document skeleton** you can use *as-is*.
It’s structured so you can **fill it incrementally** without refactoring later.

---

# 📘 Design Document

## **Intent-Driven Enterprise Assistant**

---

## 1. Document Overview

### 1.1 Purpose

This document describes the design of the **Intent-Driven Enterprise Assistant**, a system that leverages an AI agent to understand user intent, retrieve enterprise data safely, and explain system errors in plain English.

### 1.2 Scope

* Define system architecture and components
* Describe core use cases
* Specify agent, capability, and service interactions
* Outline non-functional requirements and guardrails

> **Out of Scope (for this version):**

* Deployment architecture
* CI/CD pipelines
* Performance benchmarking

---

## 2. Problem Statement

Enterprise users often struggle with:

* Searching invoices using rigid, technical query interfaces
* Understanding cryptic system and Oracle Fusion error messages
* Safely exposing enterprise data through AI systems

This project addresses these challenges using an **intent-driven agent architecture**.

---

## 3. POC Objective

> **An agent that understands user intent, retrieves enterprise data safely, and explains system errors in plain English.**

---

## 4. High-Level Architecture

### 4.1 Architectural Style

* Agent-based orchestration
* Capability-driven execution
* REST-based backend integration
* Strict separation between reasoning and data access

### 4.2 High-Level Flow (Narrative)

1. User submits a request via the UI.
2. Agent determines user intent using an LLM.
3. Agent invokes the appropriate capability.
4. Backend services query Oracle Fusion safely.
5. Results or explanations are returned to the user.

---

## 5. System Components & Responsibilities

*(Reference this section to the already finalized component document)*

### 5.1 Agent Console (UI)

### 5.2 Intent-Driven Enterprise Assistant (Agent)

### 5.3 Reasoning Engine (LLM)

### 5.4 Invoice Capability Layer

* Invoice Capability (MCP)
* Oracle Data Access Client

### 5.5 Oracle Invoice Domain Service

### 5.6 Fusion Query Builder Adapter

### 5.7 Oracle Fusion REST API

### 5.8 Oracle Database

---

## 6. Use Cases

This section describes the primary use cases supported by the **Intent-Driven Enterprise Assistant** in the current POC phase.

---

### 6.1 Invoice Search

#### 6.1.1 Description

Enable users to search for invoices using natural language without requiring knowledge of Oracle Fusion query syntax or technical parameters.

The agent interprets user intent, constructs a structured search request, and retrieves invoice data securely from Oracle Fusion via REST services.

---

#### 6.1.2 Actors

* Business User
* Support / Operations User

---

#### 6.1.3 Trigger

User submits a natural language query requesting invoice information.

---

#### 6.1.4 Supported Search Scenarios

##### a) Search by Invoice Number (Exact or Partial)

**Example Inputs:**

* “Find invoice ERS-5074”
* “Search invoices starting with ERS”
* “Show invoices containing 182311”

**System Behavior:**

* Detect invoice number search intent
* Perform exact or partial match based on input
* Return matching invoice records

---

##### b) Search by Multiple Invoice Numbers

**Example Inputs:**

* “Get details for invoices ERS-1001 and ERS-1002”
* “Show invoices INV123, INV124, INV125”

**System Behavior:**

* Extract multiple invoice identifiers
* Execute a batched search request
* Return consolidated results

---

##### c) Search by Supplier Name

**Example Inputs:**

* “Show invoices from supplier Lee”
* “Find invoices for ABC Corporation”

**System Behavior:**

* Detect supplier-based search
* Perform partial match on supplier name
* Return related invoices

---

##### d) Search by Supplier Number

**Example Inputs:**

* “Find invoices for supplier number 1252”
* “Show invoices linked to supplier 9087”

**System Behavior:**

* Identify supplier number
* Perform lookup using supplier identifier
* Return matching invoices

---

##### e) Multi-Field Combined Search

**Example Inputs:**

* “Find invoices with ERS from supplier Lee”
* “Search invoice 493527 for supplier 1252”

**System Behavior:**

* Combine multiple search criteria
* Execute controlled multi-field query
* Enforce backend guardrails (limits, validation)

---

#### 6.1.5 Preconditions

* User is authenticated
* Oracle Fusion REST service is available

---

#### 6.1.6 Postconditions

* Matching invoice data is returned to the user
* Results are formatted for readability
* No raw Fusion query syntax is exposed

---

#### 6.1.7 Alternate Flows

* No results found → User receives a clear “No matching invoices found” message
* Invalid input → User is prompted to refine search criteria
* Backend error → Error is routed to the error handling flow

---

### 6.2 Error Explanation

#### 6.2.1 Description

Enable users to understand technical system and Oracle Fusion error messages through plain-English explanations generated by the agent.

---

#### 6.2.2 Actors

* Business User
* Support / Operations User
* Developer (optional)

---

#### 6.2.3 Trigger

User submits a technical error message for explanation.

---

#### 6.2.4 Example Inputs

* Oracle Fusion REST error responses
* HTTP error messages
* Application-level exception text

---

#### 6.2.5 System Behavior

* Detect error explanation intent
* Analyze error message using the LLM
* Generate:

  * Human-readable explanation
  * Possible cause
  * Suggested next steps (if applicable)

---

#### 6.2.6 Preconditions

* User provides a non-empty error message

---

#### 6.2.7 Postconditions

* User receives a clear and simplified explanation
* No backend systems are invoked

---

#### 6.2.8 Alternate Flows

* Unrecognized error → General explanation with guidance
* Sensitive content → Sanitized response

---

### 6.3 Out-of-Scope Use Cases (Current Phase)

The following are explicitly out of scope for the current POC:

* Invoice creation or modification
* Payment processing
* Approval workflows
* Data updates in Oracle
* Cross-module reporting

---

### 6.4 Future Use Case Expansion (Informational)

This design allows future extension to:

* Payment and receipt search
* Supplier insights
* Audit and compliance explanations
* Enterprise-wide error analysis

---

## 7. Agent Design

### 7.1 Intent Classification

* INVOICE_SEARCH
* ERROR_EXPLANATION

### 7.2 Intent Normalization

* Free-text → structured intent model

### 7.3 Decision Logic

* When to call backend services
* When to rely on LLM-only responses

---

## 8. Capability Design

### 8.1 Invoice Capability (MCP)

* Inputs
* Validation rules
* Supported operations

### 8.2 Error Explanation Capability

* Input format
* Output structure

---

## 9. Backend Service Design

### 9.1 Oracle Invoice Domain Service

* API responsibilities
* Security boundaries
* Pagination and limits

### 9.2 Fusion Query Builder Adapter

* Supported fields
* Query construction rules
* Guardrails (min length, allow-list)

---

## 10. API Contracts (Draft)

### 10.1 Invoice Search API

* Request schema
* Response schema
* Error responses

### 10.2 Error Response Format

* Error codes
* Technical message
* User-facing hint

---

## 11. Error Handling Strategy

### 11.1 Backend Errors

### 11.2 Agent-Level Errors

### 11.3 User-Facing Error Messages

---

## 12. Security Considerations

* Authentication and authorization
* Input validation
* Prevention of query abuse
* LLM safety boundaries

---

## 13. Non-Functional Requirements

* Performance
* Scalability
* Reliability
* Observability
* Auditability

---

## 14. Assumptions & Constraints

* Oracle Fusion REST API behavior
* Partial search limitations
* LLM non-determinism

---

## 15. Future Enhancements

* Additional enterprise domains (payments, suppliers)
* Capability registry
* Caching layer
* Role-based access control
* Analytics and insights

---

## 16. Open Questions

* Rate limits for Fusion REST
* Maximum search result size
* Error taxonomy standardization

---

## 17. Appendix

* Glossary
* Acronyms
* References

---
