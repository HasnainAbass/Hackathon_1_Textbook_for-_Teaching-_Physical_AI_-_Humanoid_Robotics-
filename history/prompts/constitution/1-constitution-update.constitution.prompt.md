---
id: 1
title: constitution-update
stage: constitution
date: 2025-12-22
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user: user
command: /sp.constitution
labels: [constitution,spec-kit-plus,ai-book,rag-chatbot]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
---

## Prompt

Project: AI-Spec–Driven Book with Embedded RAG Chatbot

Purpose:
Create a unified, production-ready technical book using AI-assisted specification-driven development, published via Docusaurus and GitHub Pages, and enhanced with an integrated Retrieval-Augmented Generation (RAG) chatbot that can answer questions about the book's content.

Target audience:
- Intermediate to advanced developers
- AI/LLM practitioners
- Software engineering students and educators
- Technical readers with basic familiarity with web and backend concepts

Core context-limited answering)
- Architecture requirements:
  - OpenAI Agents / ChatKit SDKs for orchestration
  - FastAPI backend
  - Neon Serverless Postgres for metadata/state
  - Qdrant Cloud (Free Tier) for vector storage
- Clear separation between:
  - Retrieval
  - Prompt construction
  - Generatprinciples:
- Specification-first development (Spec-Kit Plus as the source of truth)
- Accuracy and faithfulness to written book content
- Clarity and pedagogical progression
- Reproducibility of setup, code, and deployment
- Modularity (book, backend, database, vector store, and chatbot loosely coupled)
- Security and responsible AI usage

Authoring standards (Book):
- Content authored using Docusaurus (Markdown/MDX)
- Clear chapter hierarchy with learning objectives
- Code examples must be runnable or clearly annotated
- Diagrams described textually where images are not used
- Progressive complexity from fundamentals to advanced topics
- Consistent terminology throughout the book

AI & Spec standards:
- Spec-Kit Plus drives structure, scope, and constraints
- Claude Code used for:
  - Drafting chapters
  - Generating and refining code
  - Maintaining consistency with specs
- Specs must be explicit, testable, and version-controlled
- No undocumented assumptions outside the spec

RAG Chatbot standards:
- Chation
- Hallucination minimization:
  - If answer not found in retrieved context, respond with "Not found in the book content"

Citation & traceability:
- Internal references between chapters encouraged
- All chatbot answers must be traceable to retrieved chunks
- Chunk metadata must include:
  - Chapter
  - Section
  - Source file
- No external knowledge injection unless explicitly labeled

Constraints:
- Deployment target: GitHub Pages
- Backend API must be stateless where possible
- Free-tier compatible infrastructure only
- Clear environment setup instructions
- No proprietary or paid dependencies beyond specified tools

Quality gates:
- Book builds successfully with Docusaurus
- GitHub Pages deployment passes without errors
- RAG chatbot:
  - Retrieves relevant chunks correctly
  - Answers based only on retrieved context
  - Correctly handles user-selected text queries
- API endpoints documented and testable
- Spec and implementation alignment verified

Success criteria:
- Fully navigable published book
- Embedded chatbot operational within the book UI
- Users can:
  - Ask general questions about the book
  - Ask questions limited to selected text
- No hallucinated answers
- Clean repository structure aligned with Spec-Kit Plus
- Project can be cloned and reproduced by a third party

## Response snapshot

Created a new constitution file at .specify/memory/constitution.md that defines the core principles, authoring standards, development workflow, and governance for the AI-Spec-Driven Book with Embedded RAG Chatbot project. The constitution includes principles for specification-first development, accuracy and faithfulness to content, clarity and pedagogical progression, reproducibility and modularity, security and responsible AI usage, and free-tier infrastructure compatibility.

## Outcome

- ✅ Impact: Created initial project constitution with 6 core principles and governance structure
- 🧪 Tests: No tests run yet, constitution is a documentation artifact
- 📁 Files: Updated .specify/memory/constitution.md with complete constitution content
- 🔁 Next prompts: Plan and implement the book structure and RAG chatbot components
- 🧠 Reflection: Constitution successfully captures all specified project principles and requirements

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Not applicable
- Next experiment (smallest change to try): Begin implementing the book structure based on the constitution