<!-- SYNC IMPACT REPORT
Version change: 1.0.0 → 1.0.1 (minor update with architecture details and citation standards)
Modified principles: RAG System Architecture (expanded with architecture requirements)
Added sections: Architecture Requirements section, Citation & Traceability section
Removed sections: none
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ updated
  - .specify/templates/spec-template.md ✅ updated
  - .specify/templates/tasks-template.md ✅ updated
  - .specify/templates/commands/*.md ⚠ pending
Runtime docs: README.md ⚠ pending
Follow-up TODOs: none
-->

# AI-Spec–Driven Book with Embedded RAG Chatbot Constitution

## Core Principles

### Specification-First Development
Every feature and component must be defined in the specification before implementation begins. The Spec-Kit Plus serves as the single source of truth for all requirements, constraints, and acceptance criteria. No code should be written without a corresponding specification entry.

### Accuracy and Faithfulness to Content
All book content and chatbot responses must be faithful to the written material. The RAG system must not generate hallucinated information. When the answer is not found in the retrieved context, the system must respond with "Not found in the book content".

### Clarity and Pedagogical Progression
Content must be organized with clear learning objectives, progressive complexity from fundamentals to advanced topics, and consistent terminology. Code examples must be runnable or clearly annotated as conceptual.

### Reproducibility and Modularity
The entire system must be reproducible by a third party with clear environment setup instructions. Components (book, backend, database, vector store, and chatbot) must be loosely coupled with clear interfaces.

### Security and Responsible AI Usage
All AI interactions must follow responsible usage guidelines, with proper input validation, output sanitization, and adherence to privacy requirements. Authentication and authorization mechanisms must be properly implemented where needed.

### Free-Tier Infrastructure Compatibility
All infrastructure components must be compatible with free-tier services: GitHub Pages for deployment, Neon Serverless Postgres for metadata, and Qdrant Cloud (Free Tier) for vector storage.

## Architecture Requirements

### System Components
- OpenAI Agents / ChatKit SDKs for orchestration
- FastAPI backend
- Neon Serverless Postgres for metadata/state
- Qdrant Cloud (Free Tier) for vector storage

### Architecture Standards
- Clear separation between retrieval, prompt construction, and generation
- Hallucination minimization: If answer not found in retrieved context, respond with "Not found in the book content"
- Stateless backend API where possible
- Proper error handling and graceful degradation strategies

## Authoring and Technical Standards

### Docusaurus-Based Content Creation
All book content must be authored using Docusaurus (Markdown/MDX format) with clear chapter hierarchy, internal references between chapters, and proper metadata for navigation and SEO.

### Citation & Traceability
- Internal references between chapters encouraged
- All chatbot answers must be traceable to retrieved chunks
- Chunk metadata must include: Chapter, Section, Source file
- No external knowledge injection unless explicitly labeled
- Users can ask general questions about the book and questions limited to selected text

## Development Workflow

### Quality Gates
- Book must build successfully with Docusaurus
- GitHub Pages deployment must pass without errors
- RAG chatbot must retrieve relevant chunks correctly
- Answers must be based only on retrieved context
- Correctly handles user-selected text queries
- API endpoints documented and testable
- Spec and implementation must maintain alignment

### Testing Requirements
- Unit tests for all backend components
- Integration tests for RAG pipeline
- End-to-end tests for chatbot functionality
- Build verification for Docusaurus site

### Code Standards
- All code must be version-controlled
- No undocumented assumptions outside the specification
- Claude Code must be used for drafting chapters, generating code, and maintaining consistency
- Proper error handling and graceful degradation strategies

## Governance

The constitution serves as the governing document that supersedes all other development practices. All pull requests and reviews must verify compliance with these principles. Any changes to the constitution require formal amendment procedures with proper documentation and approval. All team members must follow these principles and refer to this constitution for guidance on project decisions.

**Version**: 1.0.1 | **Ratified**: 2025-12-22 | **Last Amended**: 2025-12-22