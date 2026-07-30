# System Design: Optera Document AI Pipeline

## What We Built

We designed a pipeline optimized for cost-efficiency without sacrificing accuracy.

**The Naive Baseline vs. Optimized Route:**
Instead of sending every image to a Heavy Vision LLM (like Gemini 2.5 Flash), which is expensive and slow, we decouple OCR from the extraction logic. 

**Pipeline Stages:**

```mermaid
flowchart TD
    A[Input Image] --> B[Preprocessing (OpenCV)]
    B --> C[OCR & Table Layout (PaddleOCR + PPStructure)]
    C --> D{Density Check}
    D -- < 20 chars --> E[Reject: Non-document (Free)]
    D -- >= 20 chars --> F[Truncate OCR & LLM Routing]
    F -- Invoice / Meter --> G[gpt-oss-120b Schema Parsing (Cheap)]
    F -- Handwritten / Mechanic --> H[Vision LLM Extraction (Fallback)]
    F -- Garbage Text --> I[Reject: Non-document (Cheap)]
    G --> J[Structured JSON Output]
    H --> J
```

1. **Preprocessing (OpenCV):** Resizing and grayscaling to ensure the OCR engine gets clean input.
2. **Text Extraction (PaddleOCR & PPStructure):** We use PaddleOCR as the unified engine to extract raw text, and PPStructure to extract tables as raw HTML strings.
3. **Density Check (Zero-cost filtering):** We measure text density. Tyres, batteries, and empty walls usually return fewer than 20 characters and are rejected instantly for $0.00.
4. **LLM Routing (Truncated):** The OCR string is truncated to ~1000 characters to cap tokens. A small, fast LLM (`gpt-oss-120b` via OpenRouter) determines the document category based on this truncated text.
5. **Schema Parsing / Vision Fallback:** Standard text is sent to the LLM to map into strictly validated Pydantic models. For messy cursive (Mechanic Logs), the Pipeline gracefully falls back to the heavy Vision LLM (`gpt-4o-mini`) to read the original image.

## Detailed Execution Workflow

```mermaid
sequenceDiagram
    participant Client
    participant Main as process_image
    participant OCR as PaddleOCR
    participant Filter as Density Filter
    participant Router as LLM Router
    participant TextLLM as Text Extractor
    participant VisionLLM as Vision Extractor

    Client->>Main: Submit Document
    alt Baseline Mode
        Main->>VisionLLM: Direct to Heavy Vision Model
        VisionLLM-->>Main: Structured JSON
    else Optimized Mode
        Main->>OCR: Extract text (PaddleOCR)
        OCR-->>Main: Raw Text string
        Main->>Filter: Check text length
        alt length < 20 chars
            Filter-->>Main: Reject (Code Only)
            Main-->>Client: Error (Non-document)
        else
            Main->>Router: Classify doc type (Truncated Text)
            Router-->>Main: doc_type (invoice, mechanic_log, non_document)
            alt non_document
                Main-->>Client: Error (Rejected by router)
            else invoice / meter_reading
                Main->>TextLLM: Parse with Text Model
                TextLLM-->>Main: Validated JSON
                alt JSON is empty/null
                    Main->>VisionLLM: Fallback safety net (Universal Vision Parser)
                    VisionLLM-->>Main: Validated JSON (Handwritten/Mechanic)
                end
                Main-->>Client: Final Structured Data
            else mechanic_log / handwritten
                Main->>VisionLLM: Parse original image with Vision Model
                VisionLLM-->>Main: Validated JSON
                Main-->>Client: Final Structured Data
            end
        end
    end
```

## Cost Reduction Strategy
- **Caching/State:** The PaddleOCR and PPStructure models are instantiated as singletons so they are not reloaded per image.
- **Trimming:** By extracting the text/HTML locally first, we only send raw text to the LLM instead of high-resolution images, heavily slashing token costs.
- **Routing:** Filtering out invalid documents at the OCR stage guarantees that zero LLM tokens are wasted on non-documents.

## Where It Breaks
- **Highly Complex Layouts with Zero Text Structure:** If PaddleOCR completely fails to recognize text in a very messy document, the density filter will falsely reject it as a non-document.
- **Extremely Heavy Workloads in Python:** Since PaddleOCR is running on a CPU in the Docker container, processing 10,000s of images sequentially will be slow.

## What We Would Do With Another Week
- **Batching:** Process images in parallel using multiprocessing and batched OCR requests to max out CPU/GPU utilization.
- **Vector DB for Few-Shot Prompts:** Integrate ChromaDB or Pinecone to provide dynamic few-shot examples to the LLM based on the vendor or document style to improve JSON extraction accuracy.
- **Regex Caching for Meters:** Expand the regex engine for meter readings so that the LLM is completely bypassed for 99% of meter reads.
