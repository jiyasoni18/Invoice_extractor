# Optera Document AI Pipeline

A cost-optimized, routing-based Document AI pipeline that converts raw images of varying types into clean, structured JSON while avoiding unnecessary API calls.

## How to Run

1. Place your images in the `input/` folder.
2. Ensure you have your OpenRouter API key set:
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```
3. Run the pipeline with Docker:
   ```bash
   docker build -t optera-doc-ai .
   docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output -v $(pwd)/logs:/app/logs -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY optera-doc-ai
   ```

Outputs will be saved in `output/` as JSON files. Cost and timing statistics are appended to `logs/cost_log.csv`.

## Architecture
- **Preprocessing**: OpenCV is used to normalize the image.
- **OCR Engine**: PaddleOCR runs once per image to extract text and bounding boxes.
- **Density Check (Filtering)**: If the OCR yields very few alphanumeric characters, the image is immediately rejected (zero LLM cost).
- **Classification**: A small LLM classifies the text into a document type.
- **Extraction**: The text is parsed into a structured Pydantic schema using the LLM and validated.
