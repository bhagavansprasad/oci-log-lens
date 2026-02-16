# AI-Powered Log Deduplication System

## Overview

This project implements an AI-driven semantic deduplication system for issue logs stored in Oracle ATP 26ai.  
The system prevents duplicate Jira issue creation by automatically matching new logs against historical records using normalization, vector search, and LLM-based reasoning.

---

## Table of Contents

- [Problem Definition](#problem-definition)
- [Log Data Characteristics and Constraints](#log-data-characteristics-and-constraints)
- [Storage Model](#storage-model)
- [Role of LLM (Gemini)](#role-of-llm-gemini)
- [System Components](#system-components)
- [Pipeline 1 — Data Ingestion (Write Path)](#pipeline-1--data-ingestion-write-path)
- [Pipeline 2 — Search (Read Path / Deduplication Check)](#pipeline-2--search-read-path--deduplication-check)
- [Open Points](#open-points)

---

## Problem Definition

Oracle ATP 26ai maintains a repository of issue logs mapped to Jira IDs.  
When a new issue log is reported, there is currently no automated way to determine whether the same or a similar issue has already been logged.

### Current Challenges

- Duplicate Jira issues are created  
- Manual triage effort increases  
- Issue tracking becomes inefficient  

### Required Capability

The system must automatically process every new log, normalize its structure, and perform a semantic similarity check against historical logs to identify related or duplicate issues and return the corresponding Jira ID with a similarity score.

---

## Log Data Characteristics and Constraints

- Each log file represents a single workflow execution.  
- Each error log file results in one record in the database.  
- Log files are always provided in valid JSON format.  
- Each log file contains a list of JSON objects.  
- The structure of the JSON objects is not uniform and may vary between log files (different keys and nesting).

### Implication

Because the log structure is not fixed, the system must normalize the incoming log into a consistent schema before:

- Storing it in the database  
- Generating embeddings  
- Performing semantic search  

---

## Storage Model

For each processed log, the system stores the following in **Oracle 26ai VectorDB**:

1. **Original Log**  
   The raw log file in its native JSON format for traceability and reprocessing.

2. **Normalized Log**  
   A structured and consistent representation of the log generated after normalization.

3. **Vector Embeddings**  
   Embeddings generated from selected critical fields of the normalized log to support semantic similarity search.

### Purpose

This storage model enables efficient semantic matching of new logs against historical records while preserving the original data for audit and future reprocessing.

---

## Role of LLM (Gemini)

The LLM performs three primary functions:

1. **Normalization**  
   Converts heterogeneous logs into a consistent structure.

2. **Embedding Generation**

3. **Final Semantic Reasoning**  
   - Executed after vector search  
   - Determines actual similarity and ranking  

> This is not a pure vector search system.  
> It follows a **Vector Search + LLM Re-ranking (RAG pattern)** approach.

---

## System Components

- **UI / User Interface**
- **FASTAPI REST Layer**
- **LLM (Gemini Models)**
- **Oracle 26ai Vector Database**

---

## Pipeline 1 — Data Ingestion (Write Path)

**Purpose:** Store logs in a searchable semantic format.

### Flow

1. User submits raw log  
2. LLM normalizes it into a common schema  
3. LLM generates embeddings  
4. Database stores:
   - Raw log  
   - Normalized log  
   - Embeddings  

**Result:** Future logs can be compared semantically.

---

## Pipeline 2 — Search (Read Path / Deduplication Check)

**Purpose:** Determine whether a similar issue has already been reported.

### Flow

1. New log is submitted  
2. Log is normalized  
3. Embeddings are generated  
4. Vector search retrieves Top N candidates  
5. LLM receives:
   - New normalized log  
   - Top N similar logs  
6. LLM determines:
   - Similarity  
   - Ranking  
   - Best match  

**Final Output:** The user receives the most similar historical issues along with similarity scores and corresponding Jira IDs.

---

## Open Points

The following design decisions require further clarification:

1. What defines:
   - Error log vs informational log?

2. What is the normalized schema?

3. Which fields are used for embeddings?

4. Should Jira ID be stored in vector metadata?

5. What is the expected output format?
   - Match  
   - Duplicate  
   - Percentage similarity  

6. What is the similarity threshold for automatic deduplication?

---

## Architecture Summary

This system implements an AI-powered semantic deduplication platform using:

- Log normalization via LLM  
- Vector embeddings stored in Oracle 26ai  
- Semantic similarity search  
- LLM-based re-ranking and reasoning  

The result is a scalable, intelligent issue-matching system that reduces duplicate Jira creation and manual triage effort.


Sure! Here's what you're asking:

---

### Normalization 
- You have log files that are **valid JSON** but with **inconsistent structures** across different workflows
- You need to normalize them before storing into VectorDB

### Normalization  3 Concerns
> *"Help me design a normalization process that is controlled but also adaptable — and let's base it on real log samples rather than theory."*

**1. "How do I normalize?"**
You need a concrete technique/process to normalize logs that have varying keys, nesting, and structure into a consistent format.

**2. "I can't leave it entirely to the LLM"**
You want **control over the normalization process** — not just blindly trust the LLM to decide what's important and what's not.

**3. "But I also can't understand every workflow"**
As an AI developer, you **don't have deep domain knowledge** of every workflow that generates these logs — so you can't manually define rules for everything either.


