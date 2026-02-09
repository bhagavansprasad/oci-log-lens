
---

# 📦 Component Inventory – Invoice Search POC

## 1️⃣ Agent Console (UI Layer)

**Type:** Web UI (Streamlit)

### Responsibilities

* Accept natural language input from users
* Allow users to:

  * Search invoices (single / multiple / partial)
  * Paste technical error messages
* Display formatted results and explanations

### Non-Responsibilities

* No business logic
* No query construction
* No Oracle awareness

---

## 2️⃣ Invoice Intelligence Agent

**Type:** AI Orchestration Component

### Responsibilities

* Act as the **central coordinator**
* Route user requests to appropriate capabilities
* Decide:

  * Search invoices
  * Explain error messages
* Aggregate responses into a final user-friendly answer

### Non-Responsibilities

* No direct REST calls
* No Fusion query syntax
* No Oracle-specific logic

---

## 3️⃣ Reasoning Engine (LLM)

**Type:** Large Language Model

### Responsibilities

* Understand user intent
* Classify requests into intents:

  * `INVOICE_SEARCH`
  * `ERROR_EXPLANATION`
* Normalize free-text input into structured intent data
* Convert technical error messages into human-readable explanations

### Non-Responsibilities

* No data access
* No REST calls
* No query string construction
* No security enforcement

---

## 4️⃣ Invoice Capability Layer

**Type:** Agent Capability / Tooling Layer

This is a **logical layer** composed of two internal components.

---

### 4.1 Invoice Capability (MCP)

**Type:** Capability / Tool Adapter

### Responsibilities

* Expose invoice-related capabilities to the Agent
* Validate structured input from the Agent
* Enforce business rules:

  * Allowed search fields
  * Combination rules
  * Input sanity checks
* Call downstream domain APIs

### Non-Responsibilities

* No Fusion query syntax
* No HTTP handling
* No Oracle-specific logic

---

### 4.2 Oracle Data Access Client

**Type:** REST Client

### Responsibilities

* Handle HTTP communication
* Manage:

  * Authentication
  * Headers
  * Compression
  * Retries
  * Timeouts
* Invoke backend REST endpoints

### Non-Responsibilities

* No business logic
* No query construction
* No intent handling

---

## 5️⃣ Oracle Invoice Domain Service

**Type:** Backend REST Service

### Responsibilities

* Expose invoice-related domain APIs (e.g., search)
* Act as the **only gateway** to Oracle Fusion REST
* Validate incoming requests
* Enforce:

  * Security
  * Result limits
  * Pagination
* Return structured JSON responses

### Non-Responsibilities

* No UI concerns
* No LLM interaction

---

## 6️⃣ Fusion Query Builder Adapter

**Type:** Internal Backend Adapter

### Responsibilities

* Translate domain-level search criteria into **Fusion REST query syntax**
* Build safe `q=` expressions using:

  * `LIKE`
  * `OR`
  * `to_char(InvoiceId)` where required
* Enforce guardrails:

  * Allowed fields
  * Minimum search length
  * Controlled query patterns

### Non-Responsibilities

* No HTTP handling
* No Agent or LLM awareness

---

## 7️⃣ Oracle Fusion REST API

**Type:** External Enterprise API

### Responsibilities

* Execute invoice queries
* Apply Oracle Fusion business logic
* Return invoice data or errors

### Non-Responsibilities

* No AI logic
* No intent understanding
* No user interaction

---

## 8️⃣ Oracle Database

**Type:** Enterprise Data Store

### Responsibilities

* Persist invoice data
* Support Fusion application logic

---

# 🔗 Component Relationship Summary

```
Agent Console (UI)
        ↓
Invoice Intelligence Agent
        ↓
Reasoning Engine (LLM)
        ↓
Invoice Capability Layer
   ├── Invoice Capability (MCP)
   └── Oracle Data Access Client
        ↓
Oracle Invoice Domain Service
        ↓
Fusion Query Builder Adapter
        ↓
Oracle Fusion REST API
        ↓
Oracle Database
```

